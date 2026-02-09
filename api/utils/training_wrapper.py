"""
训练任务包装器
用于调用原始的train.py脚本，不修改其逻辑

关键设计：
- 训练子进程使用 start_new_session=True 脱离 API worker 进程组，
  确保 gunicorn worker 重启不会影响训练进程。
- 日志直接写入文件（而非通过 stdout 管道），避免管道断裂导致 SIGPIPE。
- PID 持久化到文件，即使 worker 重启也能追踪训练进程状态。
"""
import os
import sys
import json
import signal
import toml
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# 日志和PID文件目录
LOG_DIR = Path("/tmp/diffusion_pipe_logs")
PID_DIR = Path("/tmp/diffusion_pipe_pids")


class TrainingWrapper:
    """训练任务包装器"""

    def __init__(self):
        """初始化训练包装器"""
        self.project_root = Path(__file__).parent.parent.parent
        self.train_script = self.project_root / "train.py"
        # 内存中的进程引用（仅在当前 worker 生命周期内有效）
        self.processes: Dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()
        # 确保目录存在
        LOG_DIR.mkdir(exist_ok=True)
        PID_DIR.mkdir(exist_ok=True)

    def start_training(
        self,
        task_id: str,
        gpu_id: int,
        model_type: str,
        config: Dict[str, Any],
        dataset: Optional[Dict[str, Any]] = None,
        output_dir: Optional[Path] = None
    ) -> bool:
        """
        启动训练任务

        Args:
            task_id: 任务ID
            gpu_id: GPU ID
            model_type: 模型类型
            config: 训练配置
            dataset: 数据集配置
            output_dir: 输出目录

        Returns:
            是否成功启动
        """
        try:
            # 生成配置文件
            config_file = self._generate_config_file(
                task_id=task_id,
                gpu_id=gpu_id,
                model_type=model_type,
                config=config,
                dataset=dataset,
                output_dir=output_dir
            )

            # 生成随机端口避免冲突
            import random
            master_port = 29500 + random.randint(0, 99)
            
            # 构建命令行
            cmd = self._build_command(
                config_file=config_file,
                gpu_id=gpu_id,
                master_port=master_port
            )

            # 设置环境变量
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["NCCL_P2P_DISABLE"] = "1"
            env["NCCL_IB_DISABLE"] = "1"
            env["MASTER_PORT"] = str(master_port)
            # 添加项目根目录到PYTHONPATH确保模块导入正确（优先于ComfyUI的utils）
            project_root_str = str(self.project_root)
            comfyui_path = str(self.project_root / "submodules" / "ComfyUI")
            pythonpath_parts = [project_root_str, comfyui_path]
            if "PYTHONPATH" in env:
                pythonpath_parts.append(env["PYTHONPATH"])
            env["PYTHONPATH"] = ":".join(pythonpath_parts)

            # 日志直接写入文件，不通过 stdout 管道
            log_file = LOG_DIR / f"{task_id}.log"
            log_f = open(log_file, 'w', encoding='utf-8')

            # 启动训练进程
            # start_new_session=True: 创建新的进程会话，脱离 gunicorn worker 进程组
            # 这样 worker 被回收/重启时不会影响训练进程
            process = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                cwd=self.project_root,
                env=env,
                start_new_session=True,  # 关键：脱离 worker 进程组
            )

            # 记录进程到内存
            with self._lock:
                self.processes[task_id] = process

            # 持久化 PID 到文件（跨 worker 重启可追踪）
            self._save_pid(task_id, process.pid)

            # 启动后台监控线程：等待进程结束后关闭日志文件句柄
            threading.Thread(
                target=self._monitor_process,
                args=(task_id, process, log_f),
                daemon=True
            ).start()

            print(f"[{task_id}] 训练进程已启动 PID={process.pid}, 日志: {log_file}")
            return True

        except Exception as e:
            import traceback
            print(f"启动训练任务失败: {e}")
            traceback.print_exc()
            return False

    def stop_training(self, task_id: str, force: bool = False) -> bool:
        """
        停止训练任务

        Args:
            task_id: 任务ID
            force: 是否强制停止

        Returns:
            是否成功停止
        """
        # 优先从内存中获取进程引用
        with self._lock:
            process = self.processes.get(task_id)

        if process:
            try:
                if force:
                    # 杀死整个进程组（包括 deepspeed 子进程）
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                else:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass  # 进程已结束
            except Exception as e:
                print(f"停止训练任务失败: {e}")
                return False
            finally:
                with self._lock:
                    self.processes.pop(task_id, None)
                self._remove_pid(task_id)
            return True

        # 内存中没有进程引用（worker 可能已重启），从 PID 文件恢复
        pid = self._load_pid(task_id)
        if pid:
            try:
                if force:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                else:
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
            except ProcessLookupError:
                pass  # 进程已结束
            except Exception as e:
                print(f"通过PID停止训练任务失败: {e}")
                return False
            finally:
                self._remove_pid(task_id)
            return True

        return False

    def get_training_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        获取训练状态

        Args:
            task_id: 任务ID

        Returns:
            训练状态信息
        """
        # 优先从内存中获取
        with self._lock:
            process = self.processes.get(task_id)

        if process:
            poll_result = process.poll()
            return {
                "task_id": task_id,
                "running": poll_result is None,
                "returncode": process.returncode,
                "pid": process.pid
            }

        # 内存中没有，从 PID 文件恢复检查
        pid = self._load_pid(task_id)
        if pid:
            running = self._is_pid_running(pid)
            returncode = None
            if not running:
                # 进程已结束，清理 PID 文件
                self._remove_pid(task_id)
                returncode = -1  # 无法获取真实返回码，标记为异常退出
            return {
                "task_id": task_id,
                "running": running,
                "returncode": returncode,
                "pid": pid
            }

        return None

    # ==================== PID 持久化 ====================

    def _save_pid(self, task_id: str, pid: int):
        """保存训练进程 PID 到文件"""
        pid_file = PID_DIR / f"{task_id}.pid"
        pid_file.write_text(str(pid))

    def _load_pid(self, task_id: str) -> Optional[int]:
        """从文件加载训练进程 PID"""
        pid_file = PID_DIR / f"{task_id}.pid"
        if pid_file.exists():
            try:
                return int(pid_file.read_text().strip())
            except (ValueError, OSError):
                return None
        return None

    def _remove_pid(self, task_id: str):
        """删除 PID 文件"""
        pid_file = PID_DIR / f"{task_id}.pid"
        try:
            pid_file.unlink(missing_ok=True)
        except OSError:
            pass

    @staticmethod
    def _is_pid_running(pid: int) -> bool:
        """检查指定 PID 的进程是否仍在运行"""
        try:
            os.kill(pid, 0)  # 发送信号 0 不会杀死进程，只检查是否存在
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # 进程存在但无权限

    # ==================== 进程监控 ====================

    def _monitor_process(self, task_id: str, process: subprocess.Popen, log_f):
        """
        后台监控线程：等待进程结束后做清理

        Args:
            task_id: 任务ID
            process: 进程对象
            log_f: 日志文件句柄
        """
        try:
            process.wait()
            returncode = process.returncode
            print(f"[{task_id}] 训练进程结束, returncode={returncode}")
        except Exception as e:
            print(f"[{task_id}] 监控进程异常: {e}")
        finally:
            # 关闭日志文件句柄
            try:
                log_f.close()
            except Exception:
                pass
            # 清理内存中的进程引用
            with self._lock:
                self.processes.pop(task_id, None)
            # 注意：不在这里删除 PID 文件，留给 training_service 检查后再清理

    # ==================== 配置生成 ====================

    def _generate_config_file(
        self,
        task_id: str,
        gpu_id: int,
        model_type: str,
        config: Dict[str, Any],
        dataset: Optional[Dict[str, Any]] = None,
        output_dir: Optional[Path] = None
    ) -> Path:
        """
        生成训练配置文件

        Args:
            task_id: 任务ID
            gpu_id: GPU ID
            model_type: 模型类型
            config: 训练配置
            dataset: 数据集配置
            output_dir: 输出目录

        Returns:
            配置文件路径
        """
        # 生成配置目录
        config_dir = Path("/tmp/diffusion_pipe_configs")
        config_dir.mkdir(exist_ok=True)

        # 创建配置文件
        config_file = config_dir / f"{task_id}.toml"

        # 构建完整配置 - 参照 wan_odeo.toml 的格式
        full_config = {
            # 基础训练设置
            "output_dir": config.get("output_dir", str(output_dir)),
            "epochs": config.get("epochs", 60),
            "micro_batch_size_per_gpu": config.get("micro_batch_size_per_gpu", 1),
            "pipeline_stages": config.get("pipeline_stages", 1),
            "gradient_accumulation_steps": config.get("gradient_accumulation_steps", 1),
            "gradient_clipping": config.get("gradient_clipping", 1.0),
            "warmup_steps": config.get("warmup_steps", 20),
            
            # 必需的保存和检查点配置
            "save_every_n_epochs": config.get("save_every_n_epochs", 5),
            "checkpoint_every_n_epochs": config.get("checkpoint_every_n_epochs", 10),
            
            # 优化配置
            "activation_checkpointing": config.get("activation_checkpointing", True),
            "partition_method": config.get("partition_method", "parameters"),
            "save_dtype": config.get("save_dtype", "bfloat16"),
            "caching_batch_size": config.get("caching_batch_size", 1),
            "steps_per_print": config.get("steps_per_print", 1),
            "video_clip_mode": config.get("video_clip_mode", "single_beginning"),
        }
        
        # 断点续训配置
        resume_from_checkpoint = config.get("resume_from_checkpoint", False)
        if resume_from_checkpoint:
            full_config["resume_from_checkpoint"] = True

        # 显存优化配置 - blocks_to_swap > 0 时才添加
        blocks_to_swap = config.get("blocks_to_swap", 0)
        if blocks_to_swap > 0:
            full_config["blocks_to_swap"] = blocks_to_swap

        # 模型配置 - Wan模型
        model_config = config.get("model", {})
        full_config["model"] = {
            "type": model_type,
            "ckpt_path": model_config.get("ckpt_path", os.environ.get("MODELS_ROOT", "./pretrained_models/Wan2.1-I2V-14B-480P")),
            "dtype": model_config.get("dtype", "bfloat16"),
            "transformer_dtype": model_config.get("transformer_dtype", "float8"),
            "timestep_sample_method": model_config.get("timestep_sample_method", "uniform"),
        }

        # 适配器配置 - LoRA
        adapter_config = config.get("adapter", {})
        full_config["adapter"] = {
            "type": adapter_config.get("type", "lora"),
            "rank": adapter_config.get("rank", 32),
            "dtype": adapter_config.get("dtype", "bfloat16"),
        }

        # 优化器配置
        optimizer_config = config.get("optimizer", {})
        full_config["optimizer"] = {
            "type": optimizer_config.get("type", "adamw_optimi"),
            "lr": optimizer_config.get("lr", 5e-5),
            "betas": optimizer_config.get("betas", [0.9, 0.99]),
            "weight_decay": optimizer_config.get("weight_decay", 0.01),
            "eps": optimizer_config.get("eps", 1e-8),
        }

        # 数据集配置 - 必须写入单独的文件
        if dataset:
            # 强制确保使用正确的键名 'directory' (train.py 要求单数形式)
            if 'directories' in dataset:
                dataset['directory'] = dataset.pop('directories')
            
            # 确保有 directory 配置，如果没有则使用默认值
            if 'directory' not in dataset:
                dataset['directory'] = [{'path': os.environ.get('DATASET_PATH', './data/datasets'), 'num_repeats': 5}]
            
            # 生成数据集配置文件
            dataset_config_file = config_dir / f"{task_id}_dataset.toml"
            with open(dataset_config_file, 'w', encoding='utf-8') as f:
                toml.dump(dataset, f)
            # 在主配置中引用数据集文件路径
            full_config["dataset"] = str(dataset_config_file)
        else:
            # 使用默认的数据集配置文件
            full_config["dataset"] = os.path.join(os.path.dirname(__file__), "../../examples/wan_odeo_data.toml")

        # 写入配置文件
        with open(config_file, 'w', encoding='utf-8') as f:
            toml.dump(full_config, f)

        return config_file

    def _build_command(self, config_file: Path, gpu_id: int, master_port: int = 29500) -> list:
        """
        构建训练命令

        Args:
            config_file: 配置文件路径
            gpu_id: GPU ID
            master_port: 分布式训练端口

        Returns:
            命令列表
        """
        import shutil
        
        # 检查是否有deepspeed
        has_deepspeed = shutil.which("deepspeed") is not None

        # 构建命令
        if has_deepspeed:
            # 使用deepspeed启动训练
            cmd = [
                "deepspeed",
                f"--include=localhost:{gpu_id}",
                f"--master_port={master_port}",
                str(self.train_script),
                "--deepspeed",
                "--config", str(config_file),
                f"--master_port={master_port}"  # 同时传递给 train.py 确保分布式初始化使用正确端口
            ]
        else:
            # 回退到直接python运行
            cmd = [
                sys.executable,  # 使用当前Python解释器
                str(self.train_script),
                "--config", str(config_file)
            ]

        return cmd


# 全局实例（供 training_service 使用）
training_wrapper = TrainingWrapper()

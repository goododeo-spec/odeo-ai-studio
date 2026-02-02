"""
训练任务包装器
用于调用原始的train.py脚本，不修改其逻辑
"""
import os
import sys
import json
import toml
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

class TrainingWrapper:
    """训练任务包装器"""

    def __init__(self):
        """初始化训练包装器"""
        self.project_root = Path(__file__).parent.parent.parent
        self.train_script = self.project_root / "train.py"
        self.processes: Dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

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
            env["MASTER_PORT"] = str(master_port)  # 确保 deepspeed 内部也使用正确端口
            # 添加项目根目录到PYTHONPATH确保模块导入正确（优先于ComfyUI的utils）
            project_root_str = str(self.project_root)
            comfyui_path = str(self.project_root / "submodules" / "ComfyUI")
            # 项目根目录必须在ComfyUI之前，这样utils.common能正确导入
            pythonpath_parts = [project_root_str, comfyui_path]
            if "PYTHONPATH" in env:
                pythonpath_parts.append(env["PYTHONPATH"])
            env["PYTHONPATH"] = ":".join(pythonpath_parts)
            
            # 启动训练进程
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=self.project_root,
                env=env,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # 记录进程
            with self._lock:
                self.processes[task_id] = process

            # 启动日志线程
            threading.Thread(
                target=self._log_output,
                args=(task_id, process),
                daemon=True
            ).start()

            return True

        except Exception as e:
            print(f"启动训练任务失败: {e}")
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
        with self._lock:
            process = self.processes.get(task_id)
            if not process:
                return False

            try:
                if force:
                    process.kill()
                else:
                    process.terminate()

                # 等待进程结束
                process.wait(timeout=10)

                # 清理
                del self.processes[task_id]
                return True

            except subprocess.TimeoutExpired:
                # 强制杀死
                process.kill()
                del self.processes[task_id]
                return True
            except Exception as e:
                print(f"停止训练任务失败: {e}")
                return False

    def get_training_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        获取训练状态

        Args:
            task_id: 任务ID

        Returns:
            训练状态信息
        """
        with self._lock:
            process = self.processes.get(task_id)
            if not process:
                return None

            return {
                "task_id": task_id,
                "running": process.poll() is None,
                "returncode": process.returncode,
                "pid": process.pid
            }

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
                "--config", str(config_file)
            ]
        else:
            # 回退到直接python运行
            cmd = [
                sys.executable,  # 使用当前Python解释器
                str(self.train_script),
                "--config", str(config_file)
            ]

        return cmd

    def _log_output(self, task_id: str, process: subprocess.Popen):
        """
        记录输出

        Args:
            task_id: 任务ID
            process: 进程对象
        """
        log_file = Path(f"/tmp/diffusion_pipe_logs/{task_id}.log")
        log_file.parent.mkdir(exist_ok=True)

        with open(log_file, 'w', encoding='utf-8') as log_f:
            for line in process.stdout:
                log_f.write(line)
                log_f.flush()

                # 同时打印到控制台
                print(f"[{task_id}] {line.strip()}")

        # 等待进程结束
        process.wait()

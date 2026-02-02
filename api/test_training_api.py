#!/usr/bin/env python3
"""
训练API测试脚本
"""
import requests
import time
import json
from datetime import datetime

# API基础URL
API_BASE = "http://localhost:8080/api/v1"

def test_create_training_task():
    """测试创建训练任务"""
    print("\n=== 测试创建训练任务 ===")

    request_data = {
        "gpu_id": 0,
        "model_type": "flux",
        "description": "Flux LoRA 训练 - 测试数据集",
        "dataset": {
            "resolutions": [512],
            "enable_ar_bucket": True,
            "ar_buckets": [0.5, 0.75, 1.0],
            "frame_buckets": [1],
            "directories": [
                {
                    "path": "/mnt/disk0/train_data/test",
                    "num_repeats": 5
                }
            ]
        },
        "config": {
            "epochs": 10,
            "micro_batch_size_per_gpu": 1,
            "pipeline_stages": 1,
            "gradient_accumulation_steps": 1,
            "gradient_clipping": 1.0,
            "warmup_steps": 20,
            "model": {
                "diffusers_path": "/mnt/disk0/pretrained_models/FLUX.1-dev",
                "dtype": "bfloat16",
                "transformer_dtype": "float8"
            },
            "adapter": {
                "type": "lora",
                "rank": 16,
                "dtype": "bfloat16"
            },
            "optimizer": {
                "type": "adamw_optimi",
                "lr": 5e-5,
                "betas": [0.9, 0.99],
                "weight_decay": 0.01,
                "eps": 1e-8
            },
            "save": {
                "save_every_n_epochs": 2,
                "checkpoint_every_n_epochs": 5,
                "save_dtype": "bfloat16"
            },
            "optimization": {
                "activation_checkpointing": True,
                "partition_method": "parameters",
                "caching_batch_size": 1,
                "steps_per_print": 1,
                "video_clip_mode": "single_beginning"
            }
        }
    }

    try:
        response = requests.post(
            f"{API_BASE}/training/start",
            json=request_data
        )

        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

        if response.status_code == 201:
            task_id = response.json()['data']['task_id']
            print(f"\n✅ 任务创建成功，任务ID: {task_id}")
            return task_id
        else:
            print(f"\n❌ 任务创建失败")
            return None

    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return None

def test_list_training_tasks():
    """测试获取训练任务列表"""
    print("\n=== 测试获取训练任务列表 ===")

    try:
        response = requests.get(f"{API_BASE}/training/list")

        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

        if response.status_code == 200:
            print(f"\n✅ 任务列表获取成功")
            return response.json()['data']
        else:
            print(f"\n❌ 任务列表获取失败")
            return None

    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return None

def test_get_training_task(task_id):
    """测试获取训练任务详情"""
    print(f"\n=== 测试获取训练任务详情 (task_id: {task_id}) ===")

    try:
        response = requests.get(f"{API_BASE}/training/{task_id}")

        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

        if response.status_code == 200:
            print(f"\n✅ 任务详情获取成功")
            return response.json()['data']
        else:
            print(f"\n❌ 任务详情获取失败")
            return None

    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return None

def test_get_training_progress(task_id):
    """测试获取训练进度"""
    print(f"\n=== 测试获取训练进度 (task_id: {task_id}) ===")

    try:
        response = requests.get(f"{API_BASE}/training/{task_id}/progress")

        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

        if response.status_code == 200:
            print(f"\n✅ 训练进度获取成功")
            return response.json()['data']
        else:
            print(f"\n❌ 训练进度获取失败")
            return None

    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return None

def test_get_training_metrics(task_id):
    """测试获取训练指标历史"""
    print(f"\n=== 测试获取训练指标历史 (task_id: {task_id}) ===")

    try:
        response = requests.get(
            f"{API_BASE}/training/{task_id}/metrics",
            params={
                "metric": "loss",
                "type": "epoch",
                "limit": 10
            }
        )

        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

        if response.status_code == 200:
            print(f"\n✅ 训练指标获取成功")
            return response.json()['data']
        else:
            print(f"\n❌ 训练指标获取失败")
            return None

    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return None

def test_get_training_logs(task_id):
    """测试获取训练日志"""
    print(f"\n=== 测试获取训练日志 (task_id: {task_id}) ===")

    try:
        response = requests.get(
            f"{API_BASE}/training/{task_id}/logs",
            params={
                "type": "train",
                "tail": 20
            }
        )

        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

        if response.status_code == 200:
            print(f"\n✅ 训练日志获取成功")
            return response.json()['data']
        else:
            print(f"\n❌ 训练日志获取失败")
            return None

    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return None

def test_stop_training(task_id):
    """测试停止训练任务"""
    print(f"\n=== 测试停止训练任务 (task_id: {task_id}) ===")

    try:
        response = requests.post(
            f"{API_BASE}/training/stop/{task_id}",
            json={"force": False}
        )

        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

        if response.status_code == 200:
            print(f"\n✅ 训练任务停止成功")
            return response.json()['data']
        else:
            print(f"\n❌ 训练任务停止失败")
            return None

    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return None

def main():
    """主函数"""
    print("=" * 60)
    print("Diffusion-Pipe 训练API测试")
    print("=" * 60)

    # 检查API服务是否可用
    try:
        response = requests.get(f"{API_BASE}/gpu/status", timeout=5)
        print(f"✅ API服务已启动")
    except requests.exceptions.ConnectionError:
        print("❌ API服务未启动，请先启动API服务")
        print("   运行: python run.py")
        return
    except Exception as e:
        print(f"❌ 无法连接到API服务: {e}")
        return

    # 执行测试
    task_id = None

    # 1. 测试创建训练任务
    task_id = test_create_training_task()

    if task_id:
        # 等待一下
        print("\n⏳ 等待5秒...")
        time.sleep(5)

        # 2. 测试获取训练任务列表
        test_list_training_tasks()

        # 3. 测试获取训练任务详情
        test_get_training_task(task_id)

        # 4. 测试获取训练进度
        test_get_training_progress(task_id)

        # 5. 测试获取训练指标
        test_get_training_metrics(task_id)

        # 6. 测试获取训练日志
        test_get_training_logs(task_id)

        # 等待一下再停止
        print("\n⏳ 等待5秒...")
        time.sleep(5)

        # 7. 测试停止训练任务
        test_stop_training(task_id)

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    main()

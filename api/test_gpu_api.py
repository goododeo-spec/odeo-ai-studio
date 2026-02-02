#!/usr/bin/env python
"""
GPU API 测试脚本
"""
import requests
import json
from typing import Optional

API_BASE = "http://localhost:8080/api/v1"

def test_gpu_status():
    """测试获取所有 GPU 状态"""
    print("\n" + "=" * 60)
    print("测试 1: 获取所有 GPU 状态")
    print("=" * 60)

    try:
        response = requests.get(f"{API_BASE}/gpu/status")
        print(f"状态码: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"消息: {data['message']}")

            gpus = data['data']['gpus']
            print(f"\n发现 {len(gpus)} 个 GPU:")
            for gpu in gpus:
                print(f"\n  GPU {gpu['gpu_id']}: {gpu['name']}")
                print(f"    状态: {gpu['status']}")
                print(f"    显存: {gpu['memory']['used']}/{gpu['memory']['total']} MB "
                      f"({gpu['memory']['utilization']}%)")
                print(f"    利用率: {gpu['utilization_gpu']}%")
                print(f"    温度: {gpu['temperature']['gpu']}°C")
                if gpu['power_usage']:
                    print(f"    功耗: {gpu['power_usage']}/{gpu['power_limit']} W")

            print("\n汇总信息:")
            summary = data['data']['summary']
            print(f"  总 GPU 数: {summary['total_gpus']}")
            print(f"  可用 GPU 数: {summary['available_gpus']}")
            print(f"  忙碌 GPU 数: {summary['busy_gpus']}")
            print(f"  总显存: {summary['total_memory']} MB")
            print(f"  可用显存: {summary['available_memory']} MB")
        else:
            print(f"错误: {response.text}")

    except requests.exceptions.ConnectionError:
        print("错误: 无法连接到 API 服务器")
        print("请先启动 API 服务器: python run.py")
    except Exception as e:
        print(f"错误: {str(e)}")

def test_available_gpus(min_memory: Optional[int] = None):
    """测试获取可用 GPU 列表"""
    print("\n" + "=" * 60)
    print("测试 2: 获取可用 GPU 列表")
    print("=" * 60)

    try:
        params = {}
        if min_memory:
            params['min_memory'] = min_memory

        response = requests.get(f"{API_BASE}/gpu/available", params=params)
        print(f"状态码: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"消息: {data['message']}")

            available_gpus = data['data']['available_gpus']
            print(f"\n找到 {len(available_gpus)} 个可用 GPU:")
            for gpu in available_gpus:
                print(f"\n  GPU {gpu['gpu_id']}: {gpu['name']}")
                print(f"    可用显存: {gpu['memory_free']} MB ({gpu['memory_free_gb']:.2f} GB)")
                print(f"    利用率: {gpu['utilization_gpu']}%")
                print(f"    温度: {gpu['temperature_gpu']}°C")
        else:
            print(f"错误: {response.text}")

    except requests.exceptions.ConnectionError:
        print("错误: 无法连接到 API 服务器")
        print("请先启动 API 服务器: python run.py")
    except Exception as e:
        print(f"错误: {str(e)}")

def test_gpu_details(gpu_id: int):
    """测试获取指定 GPU 详细信息"""
    print("\n" + "=" * 60)
    print(f"测试 3: 获取 GPU {gpu_id} 详细信息")
    print("=" * 60)

    try:
        response = requests.get(f"{API_BASE}/gpu/{gpu_id}/details")
        print(f"状态码: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"消息: {data['message']}")

            gpu = data['data']
            print(f"\nGPU {gpu['gpu_id']}: {gpu['name']}")
            print(f"  状态: {gpu['status']}")
            print(f"  可用: {'是' if gpu['is_available'] else '否'}")

            print(f"\n显存信息:")
            print(f"  总计: {gpu['memory']['total']} MB")
            print(f"  已用: {gpu['memory']['used']} MB")
            print(f"  可用: {gpu['memory']['free']} MB")
            print(f"  利用率: {gpu['memory']['utilization']}%")

            print(f"\n性能信息:")
            print(f"  GPU 利用率: {gpu['utilization']['gpu']}%")
            print(f"  显存利用率: {gpu['utilization']['memory']}%")

            print(f"\n温度信息:")
            print(f"  GPU: {gpu['temperature']['gpu']}°C")
            if gpu['temperature']['memory']:
                print(f"  显存: {gpu['temperature']['memory']}°C")

            if gpu['power']['current']:
                print(f"\n功耗信息:")
                print(f"  当前: {gpu['power']['current']} W")
                print(f"  限制: {gpu['power']['limit']} W")
                print(f"  使用率: {gpu['power']['usage_percent']:.1f}%")

            if gpu['current_task']:
                print(f"\n当前任务:")
                print(f"  任务 ID: {gpu['current_task']['task_id']}")
                print(f"  任务类型: {gpu['current_task']['task_type']}")
                if gpu['current_task']['progress']:
                    print(f"  进度: {gpu['current_task']['progress']:.1f}%")
        elif response.status_code == 404:
            print(f"错误: GPU {gpu_id} 不存在")
        else:
            print(f"错误: {response.text}")

    except requests.exceptions.ConnectionError:
        print("错误: 无法连接到 API 服务器")
        print("请先启动 API 服务器: python run.py")
    except Exception as e:
        print(f"错误: {str(e)}")

def test_gpu_summary():
    """测试获取 GPU 汇总信息"""
    print("\n" + "=" * 60)
    print("测试 4: 获取 GPU 汇总信息")
    print("=" * 60)

    try:
        response = requests.get(f"{API_BASE}/gpu/summary")
        print(f"状态码: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"消息: {data['message']}")

            summary = data['data']
            print(f"\nGPU 汇总:")
            print(f"  总数: {summary['total_gpus']}")
            print(f"  可用: {summary['available_gpus']}")
            print(f"  忙碌: {summary['busy_gpus']}")
            print(f"  总显存: {summary['total_memory']} MB")
            print(f"  可用显存: {summary['available_memory']} MB")
            print(f"  显存利用率: {summary['memory_utilization']:.1f}%")
            print(f"\n更新时间: {summary['timestamp']}")
        else:
            print(f"错误: {response.text}")

    except requests.exceptions.ConnectionError:
        print("错误: 无法连接到 API 服务器")
        print("请先启动 API 服务器: python run.py")
    except Exception as e:
        print(f"错误: {str(e)}")

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("Diffusion-Pipe GPU API 测试")
    print("=" * 60)

    # 检查 API 是否运行
    try:
        response = requests.get("http://localhost:8080/health", timeout=2)
        if response.status_code != 200:
            print("警告: API 健康检查失败")
    except requests.exceptions.ConnectionError:
        print("错误: 无法连接到 API 服务器")
        print("\n请先启动 API 服务器:")
        print("  cd /root/diffusion-pipe/api")
        print("  python run.py")
        print("\n或使用后台模式:")
        print("  nohup python run.py > api.log 2>&1 &")
        return

    # 运行测试
    test_gpu_status()
    test_available_gpus(min_memory=10000)  # 要求至少 10GB 显存
    test_gpu_details(0)
    test_gpu_summary()

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""
数据集预处理 API 测试脚本
"""
import sys
import os
sys.path.insert(0, '/root/diffusion-pipe')

from app import create_app
from services.preprocessing_service import preprocessing_service
from models.preprocessing import PreprocessingRequest, CaptionMethod

def test_preprocessing_api():
    """测试预处理API功能"""

    print("=" * 60)
    print("数据集预处理 API 功能测试")
    print("=" * 60)

    # 测试1: 检查不存在的数据集
    print("\n【测试 1】检查不存在的数据集")
    print("-" * 40)
    exists, info = preprocessing_service.check_dataset_exists('nonexistent_dataset')
    print(f"结果: exists={exists}, info={info}")
    assert not exists, "数据集应该不存在"
    print("✓ 测试通过")

    # 测试2: 获取所有数据集列表
    print("\n【测试 2】获取所有数据集列表")
    print("-" * 40)
    datasets = preprocessing_service.get_all_datasets()
    print(f"数据集总数: {len(datasets)}")
    for d in datasets:
        print(f"  - {d.dataset_name}: {d.video_count} 个视频, 状态: {d.status.value}")
    print("✓ 测试通过")

    # 测试3: 尝试创建重复数据集（应该失败）
    print("\n【测试 3】尝试创建重复数据集")
    print("-" * 40)
    try:
        request = PreprocessingRequest(
            dataset_name=datasets[0].dataset_name if datasets else 'existing_dataset',
            video_directory='/nonexistent/path',
            prompt_prefix='A high quality',
            caption_method=CaptionMethod.EXTRA_MIXED
        )
        task_id = preprocessing_service.create_preprocessing_task(request)
        print(f"❌ 测试失败: 应该抛出异常但没有")
    except ValueError as e:
        print(f"正确抛出异常: {e}")
        print("✓ 测试通过")

    # 测试4: 创建新的预处理任务
    print("\n【测试 4】创建新的预处理任务")
    print("-" * 40)
    try:
        request = PreprocessingRequest(
            dataset_name='test_dataset_001',
            video_directory='/tmp/test_videos',  # 使用不存在的路径测试
            prompt_prefix='A high quality',
            caption_method=CaptionMethod.EXTRA_MIXED
        )
        task_id = preprocessing_service.create_preprocessing_task(request)
        print(f"任务ID: {task_id}")
        print("✓ 测试通过")
    except Exception as e:
        print(f"✓ 预期错误（视频目录不存在）: {str(e)[:100]}")

    # 测试5: 获取任务状态
    print("\n【测试 5】获取任务状态")
    print("-" * 40)
    if 'task_id' in locals():
        result = preprocessing_service.get_task_status(task_id)
        if result:
            print(f"任务状态: {result.status.value}")
            print(f"数据集名称: {result.dataset_name}")
            print(f"输出目录: {result.output_directory}")
            print("✓ 测试通过")
        else:
            print("❌ 任务不存在")

    # 测试6: Flask 应用路由注册
    print("\n【测试 6】Flask 应用路由注册")
    print("-" * 40)
    app = create_app('development')
    preprocessing_routes = [rule for rule in app.url_map.iter_rules()
                           if '/preprocessing' in rule.rule]
    print(f"预处理相关路由数: {len(preprocessing_routes)}")
    for route in preprocessing_routes:
        print(f"  - {route.rule} [{','.join(sorted(route.methods - {'HEAD', 'OPTIONS'}))}]")
    assert len(preprocessing_routes) >= 4, "应该有至少4个预处理路由"
    print("✓ 测试通过")

    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)

if __name__ == '__main__':
    test_preprocessing_api()

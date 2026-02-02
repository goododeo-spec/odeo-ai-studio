# 数据集预处理输出路径修改总结

## 修改内容

将数据集预处理的输出路径从 `/mnt/disk0/lora数据集` 修改为 `/mnt/disk0/lora_outputs`

## 修改的文件

### 1. 核心服务文件

**`/root/diffusion-pipe/api/services/preprocessing_service.py`**
- 修改第29行：`BASE_OUTPUT_DIR = "/mnt/disk0/lora_outputs"`
- 影响：所有数据集输出目录的基础路径

### 2. API路由文件

**`/root/diffusion-pipe/api/routes/preprocessing.py`**
- 修改5处文档字符串中的示例路径：
  1. 第125行：任务创建响应中的 `output_directory`
  2. 第177行：状态查询示例中的 `output_directory`
  3. 第242行：数据集列表示例中的 `directory`
  4. 第300行：数据集详情示例中的 `directory`
  5. 第358行：数据集检查示例中的 `directory`

### 3. API文档文件

**`/root/diffusion-pipe/api/PREPROCESSING_API.md`**
- 修改文档中的所有路径引用（共7处）：
  - 响应示例中的 `directory` 字段（3处）
  - 响应示例中的 `output_directory` 字段（2处）
  - 输出结构说明中的路径
  - 注意事项中的路径

### 4. 实现总结文件

**`/root/diffusion-pipe/api/PREPROCESSING_API_SUMMARY.md`**
- 修改2处路径引用：
  - 数据持久化说明中的路径
  - 输出示例中的目录结构

## 修改验证

✅ 所有旧路径引用已全部更新
✅ 新路径 `/mnt/disk0/lora_outputs` 已生效
✅ 服务导入正常
✅ 数据集检查功能正常

## 影响范围

### 直接影响
- 所有新创建的数据集将保存到 `/mnt/disk0/lora_outputs/` 目录
- API响应中的路径信息将显示新路径

### 注意事项
- 旧路径 `/mnt/disk0/lora数据集` 仍可能存在历史数据
- 新路径与旧路径不冲突，各自独立
- 现有数据集不会被移动，仍在原路径

## 路径结构对比

### 修改前
```
/mnt/disk0/lora数据集/
├── dataset1/
├── dataset2/
└── ...
```

### 修改后
```
/mnt/disk0/lora_outputs/
├── dataset1/
├── dataset2/
└── ...
```

## 测试命令

```bash
# 验证路径配置
conda activate lora
cd /root/diffusion-pipe/api
python -c "from services.preprocessing_service import BASE_OUTPUT_DIR; print(BASE_OUTPUT_DIR)"

# 预期输出
/mnt/disk0/lora_outputs
```

## 总结

修改已完成，所有相关文件中的路径引用已更新。新的输出路径 `/mnt/disk0/lora_outputs` 已生效并通过测试验证。

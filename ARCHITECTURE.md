# Diffusion-Pipe 项目架构文档

## 1. 项目概述

**diffusion-pipe** 是一个基于 DeepSpeed 管道并行训练的大规模扩散模型训练框架，支持 19+ 种图像和视频生成模型，包括 SDXL、Flux、LTX-Video、HunyuanVideo 等。该框架采用模块化设计，提供统一的训练接口，支持 LoRA 高效微调和全参数微调，特别适合大规模分布式训练场景。

### 核心特性

- **管道并行训练**：基于 DeepSpeed 实现模型在多 GPU 上的自动分割
- **统一训练接口**：同时支持图像和视频模型，一套代码处理多种架构
- **高效缓存系统**：SQLite + 分片文件的预缓存机制，支持 TB 级数据集
- **多精度支持**：fp8、bf16、fp16 等，降低内存占用
- **适配器训练**：原生支持 LoRA、IA3 等参数高效微调方法
- **评估系统**：内置评估数据集、时间步分位数评估、TensorBoard/W&B 日志
- **检查点机制**：完整的训练状态保存与恢复

## 2. 项目结构

```
/root/diffusion-pipe/
├── train.py                     # 主训练脚本，895 行
├── models/                      # 模型实现目录 (18+ 个模型)
│   ├── base.py                  # 基础管道类定义
│   ├── flux.py                  # Flux 模型实现
│   ├── sdxl.py                  # SDXL 模型实现
│   ├── hunyuan_video.py         # 混元视频模型
│   ├── wan/                     # Wan 模型子目录
│   │   ├── model.py             # 主模型
│   │   ├── attention.py         # 注意力机制
│   │   ├── clip.py              # CLIP 文本编码器
│   │   ├── t5.py                # T5 编码器
│   │   ├── vae2_1.py            # VAE 2.1
│   │   └── vae2_2.py            # VAE 2.2
│   └── ...                      # 其他模型 (chroma, hidream, sd3, cosmos 等)
├── utils/                       # 工具模块
│   ├── dataset.py               # 数据处理核心 (665 行)
│   ├── cache.py                 # 缓存系统
│   ├── common.py                # 通用工具
│   ├── offloading.py            # 模型块交换卸载
│   ├── patches.py               # 代码猴子补丁
│   ├── pipeline.py              # 管道工具
│   └── saver.py                 # 检查点保存
├── optimizers/                  # 优化器实现
│   ├── generic_optim.py         # 通用优化器，支持 SM/Muon (275 行)
│   ├── automagic.py             # 自动优化器
│   ├── adamw_8bit.py            # 8bit AdamW + Kahan 求和
│   └── gradient_release.py      # 梯度释放优化器
├── configs/                     # 预置配置文件
│   ├── flux_vae/                # Flux VAE 配置
│   ├── hunyuan_image/           # 混元图像配置
│   ├── qwen_image/              # Qwen 图像配置
│   └── ...                      # 其他模型配置
├── examples/                    # 示例配置文件
│   ├── main_example.toml        # 主示例配置 (带注释)
│   ├── dataset.toml             # 数据集配置示例
│   ├── wan_14b_min_vram.toml    # 低显存配置
│   └── ...                      # 其他示例
├── submodules/                  # 外部依赖子模块
│   ├── ComfyUI/                 # ComfyUI 后端 (完整副本)
│   ├── HunyuanVideo/            # 混元视频代码
│   ├── Cosmos/                  # NVIDIA Cosmos 视频模型
│   ├── Lumina_2/                # Lumina 图像模型
│   └── ...                      # 其他模型子模块
├── docs/
│   └── supported_models.md      # 支持的模型列表
└── test/                        # 测试相关
```

## 3. 核心训练流程

### 3.1 训练架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        train.py (主训练脚本)                        │
├─────────────────────────────────────────────────────────────────┤
│  1. 配置解析 (TOML) → 2. 分布式初始化 → 3. 模型工厂加载 → 4. 数据缓存 │
│                                                                    │
│  5. DeepSpeed 管道并行设置 → 6. 优化器配置 → 7. 训练循环 → 8. 评估   │
│                                                                    │
│  9. 检查点保存 → 10. 恢复机制                                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    DeepSpeed PipelineModule                      │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐      │
│  │  GPU 0    │  │  GPU 1    │  │  GPU 2    │  │  GPU 3    │      │
│  │  Layers   │  │  Layers   │  │  Layers   │  │  Layers   │      │
│  │  0-9      │  │  10-19    │  │  20-29    │  │  30-39    │      │
│  └───────────┘  └───────────┘  └───────────┘  └───────────┘      │
│        ↓             ↓             ↓             ↓                │
│  [Data Parallel Group 1]    [Data Parallel Group 2]             │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 主训练流程 (train.py 核心逻辑)

```python
def main():
    # ========== 初始化阶段 ==========
    1. 解析命令行参数和 TOML 配置文件
    2. 初始化分布式训练环境 (DeepSpeed)
    3. 动态导入模型类并实例化
    4. 加载数据集配置 (TOML)
    5. 初始化数据集管理器

    # ========== 模型加载阶段 ==========
    model.load_diffusion_model()           # 加载扩散模型权重
    if adapter_config:
        model.configure_adapter()          # 配置 LoRA 适配器
        model.load_adapter_weights()       # 加载已有适配器 (可选)

    # ========== 管道并行设置 ==========
    layers = model.to_layers()             # 将模型转换为层列表
    pipeline_model = ManualPipelineModule(
        layers=layers,
        num_stages=pipeline_stages,
        loss_fn=model.get_loss_fn()
    )

    # ========== DeepSpeed 初始化 ==========
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=pipeline_model,
        config=ds_config
    )

    # ========== 训练循环 ==========
    while step < max_steps:
        # 前向 + 反向传播
        iterator = get_data_iterator_for_step(train_dataloader, model_engine)
        loss = model_engine.train_batch(iterator)

        # 评估
        if should_eval():
            evaluate(model, eval_dataloaders)

        # 保存检查点
        if should_save():
            save_checkpoint()

        step += 1
```

### 3.3 模型工厂模式

```python
# train.py 第 303-355 行：模型类型映射
MODEL_MAP = {
    'flux': flux.FluxPipeline,
    'sdxl': sdxl.SDXLPipeline,
    'ltx-video': ltx_video.LTXVideoPipeline,
    'hunyuan-video': hunyuan_video.HunyuanVideoPipeline,
    'sdxl': sdxl.SDXLPipeline,
    'cosmos': cosmos.CosmosPipeline,
    'lumina_2': lumina_2.Lumina2Pipeline,
    'wan': wan.WanPipeline,
    'chroma': chroma.ChromaPipeline,
    'hidream': hidream.HiDreamPipeline,
    'sd3': sd3.SD3Pipeline,
    'cosmos_predict2': cosmos_predict2.CosmosPredict2Pipeline,
    'omnigen2': omnigen2.OmniGen2Pipeline,
    'qwen_image': qwen_image.QwenImagePipeline,
    'hunyuan_image': hunyuan_image.HunyuanImagePipeline,
    'auraflow': auraflow.AuraFlowPipeline,
    'z_image': z_image.ZImagePipeline,
    'hunyuan_video_15': hunyuan_video_15.HunyuanVideo15Pipeline,
}

model_type = config['model']['type']
model_class = MODEL_MAP[model_type]
model = model_class(config)  # 动态实例化
```

## 4. 模型系统架构

### 4.1 基础类层次

```
BasePipeline (base.py)
├── 抽象方法定义:
│   ├── load_diffusion_model()     # 加载扩散模型
│   ├── get_vae()                  # 获取 VAE
│   ├── get_text_encoders()        # 获取文本编码器
│   ├── configure_adapter()        # 配置 LoRA
│   ├── save_adapter()             # 保存适配器
│   ├── prepare_inputs()           # 准备输入
│   ├── to_layers()                # 转换为管道层
│   └── get_loss_fn()              # 获取损失函数
│
└── ComfyPipeline (base.py)
    ├── 继承 BasePipeline
    ├── 集成 ComfyUI 后端
    ├── VAE 编码/解码
    ├── 文本编码器集成
    └── 权重管理
```

### 4.2 典型模型实现：Flux

```python
# models/flux.py 核心结构
class FluxPipeline(BasePipeline):
    def load_diffusion_model(self):
        # 1. 加载双块和单块
        # 2. 加载 VAE
        # 3. 权重映射 (BFL → Diffusers)
        pass

    def to_layers(self):
        # 将模型转换为层列表供 DeepSpeed 分割
        return [
            self_double_blocks,    # 19 个双块
            self_single_blocks,    # 38 个单块
            self.final_layer,
            self.guidance_embedder  # 可选 bypass
        ]

    def prepare_inputs(self, inputs, timestep_quantile):
        # 1. 时间步采样
        # 2. 文本嵌入处理
        # 3. 图像/视频拼接
        # 4. 舍入到倍数
        pass
```

### 4.3 适配器系统 (LoRA)

```python
# models/base.py 第 176-204 行：LoRA 配置
def configure_adapter(self, adapter_config):
    # 1. 自动检测目标线性层
    target_linear_modules = set()
    for name, module in self.transformer.named_modules():
        if module.__class__.__name__ not in self.adapter_target_modules:
            continue
        for full_submodule_name, submodule in module.named_modules():
            if isinstance(submodule, nn.Linear):
                target_linear_modules.add(full_submodule_name)

    # 2. 创建 LoRA 配置
    peft_config = peft.LoraConfig(
        r=adapter_config['rank'],              # LoRA 秩
        lora_alpha=adapter_config['alpha'],    # 缩放 (强制等于 rank)
        lora_dropout=adapter_config['dropout'],
        bias='none',
        target_modules=list(target_linear_modules)  # 目标模块列表
    )

    # 3. 应用适配器
    self.peft_config = peft_config
    self.lora_model = peft.get_peft_model(self.transformer, peft_config)
```

### 4.4 适配器目标模块映射

| 模型 | 目标模块 |
|------|----------|
| Flux | `Linear`, `LinearGEGLU` |
| SDXL | `Attention`, `FeedForward` |
| Wan | `Attention` |
| HunyuanVideo | `Attention`, `MLP` |
| HunyuanImage | `TransformerBlock` |

## 5. 数据处理系统

### 5.1 缓存架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        缓存系统架构                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────┐                                        ┌───┐ │
│  │ metadata.db   │ ←─────────── SQLite 元数据索引             │   │ │
│  │ (指纹、索引)   │                                        │   │ │
│  └───────────────┘                                        │   │ │
│         │                                                 │   │ │
│         ↓                                                 │   │ │
│  ┌───────────────┐                                        │   │ │
│  │ shard_000.bin │ ←─────── 分片二进制文件 (1GB/片)           │   │ │
│  │ shard_001.bin │                                        │   │ │
│  │ shard_002.bin │                                        │   │ │
│  │ ...           │                                        │   │ │
│  └───────────────┘                                        │   │ │
│         │                                                 │   │ │
│         ↓                                                 │   │ │
│  ┌───────────────┐ ←─── DatasetManager 统一管理             │   │ │
│  │ Train Dataset │                                              │ │
│  └───────────────┘                                              │ │
│                                                                  │
│  缓存流程:                                                       │
│  1. 计算数据集指纹 (文件 + 配置)                                  │
│  2. 检查现有缓存匹配                                              │
│  3. 多进程并行处理数据                                            │
│  4. 递归克隆张量 (避免引用)                                        │
│  5. 批量写入分片                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 数据预处理流水线

```python
# utils/dataset.py 核心类
class PreprocessMediaFile:
    def __call__(self, spec, mask_filepath, size_bucket):
        # 1. 加载媒体文件
        if is_video:
            video = imageio.v3.imiter(filepath_or_file, fps=self.framerate)
        else:
            pil_img = Image.open(filepath_or_file)

        # 2. 透明图像处理 (RGBA → RGB)
        if pil_img.mode == 'RGBA':
            canvas = Image.new('RGB', pil_img.size, (255, 255, 255))
            canvas.alpha_composite(pil_img)
            pil_img = canvas

        # 3. 调整大小 (保持宽高比)
        height_rounded = round_to_nearest_multiple(size_bucket_height, 16)
        width_rounded = round_to_nearest_multiple(size_bucket_width, 16)
        frames_rounded = round_down_to_multiple(size_bucket_frames - 1, 4) + 1

        # 4. 转换为张量并归一化 [-1, 1]
        resized_video = torch.empty((num_frames, 3, height_rounded, width_rounded))
        for i, frame in enumerate(video):
            cropped_image = convert_crop_and_resize(frame, (width_rounded, height_rounded))
            resized_video[i] = self.pil_to_tensor(cropped_image)

        # 5. 维度转换 (num_frames, C, H, W) → (C, num_frames, H, W)
        if self.support_video:
            resized_video = torch.permute(resized_video, (1, 0, 2, 3))

        # 6. 视频剪辑提取
        if is_video:
            videos = extract_clips(resized_video, frames_rounded, self.video_clip_mode)
            return [(video, mask) for video in videos]
        else:
            return [(resized_video.squeeze(0), mask)]
```

### 5.3 视频剪辑模式

```python
# utils/base.py 第 31-52 行
def extract_clips(video, target_frames, video_clip_mode):
    frames = video.shape[1]

    if video_clip_mode == 'single_beginning':
        # 从视频开头提取指定帧数
        return [video[:, :target_frames, ...]]

    elif video_clip_mode == 'single_middle':
        # 从视频中间提取指定帧数
        start = int((frames - target_frames) / 2)
        return [video[:, start:start+target_frames, ...]]

    # elif video_clip_mode == 'multiple_overlapping':
    #     # 多个重叠剪辑 (已废弃)
    #     num_clips = ((frames - 1) // target_frames) + 1
    #     start_indices = torch.linspace(0, frames-target_frames, num_clips).int()
    #     return [video[:, i:i+target_frames, ...] for i in start_indices]
```

### 5.4 数据集管理

```python
# utils/dataset.py 第 650+ 行
class DatasetManager:
    def __init__(self, model, regenerate_cache=False, trust_cache=False):
        self.model = model
        self.regenerate_cache = regenerate_cache
        self.trust_cache = trust_cache
        self.datasets = []

    def register(self, dataset):
        """注册数据集"""
        self.datasets.append(dataset)

    def cache(self, unload_models=True):
        """缓存所有注册的数据集"""
        for dataset in self.datasets:
            self.cache_dataset(dataset, unload_models)

    def cache_dataset(self, dataset, unload_models):
        """缓存单个数据集"""
        # 1. 计算指纹
        fingerprint = self.compute_fingerprint(dataset)

        # 2. 检查现有缓存
        if not self.regenerate_cache and self.trust_cache:
            # 直接加载缓存
            self.load_from_cache(dataset)
        elif self.is_cache_valid(dataset, fingerprint):
            # 验证缓存
            self.load_from_cache(dataset)
        else:
            # 重新生成缓存
            self.generate_cache(dataset, fingerprint)

    def compute_fingerprint(self, dataset):
        """计算数据集指纹"""
        # 基于文件列表、配置、时间戳计算哈希
        pass

    def generate_cache(self, dataset, fingerprint):
        """生成缓存文件"""
        # 多进程并行处理
        with mp.Pool(NUM_PROC) as pool:
            results = pool.map(self.process_item, dataset.items)

        # 写入分片文件
        self.write_shards(results)
```

## 6. 分布式训练实现

### 6.1 DeepSpeed 管道并行

```python
# train.py 第 567-577 行：管道模块创建
num_stages = config.get('pipeline_stages', 1)
partition_method = config.get('partition_method', 'parameters')

pipeline_model = ManualPipelineModule(
    layers=layers,                              # 模型层列表
    num_stages=num_stages,                      # 管道阶段数 (= GPU 数)
    partition_method=partition_method,          # 分割方法
    manual_partition_split=partition_split,     # 手动分割点 (可选)
    loss_fn=model.get_loss_fn(),                # 损失函数
    **additional_pipeline_module_kwargs         # 其他参数
)
```

### 6.2 混合并行策略

```
场景: 4 GPUs, pipeline_stages=2

┌─────────────────────────────────────────┐
│            Data Parallel Group 1         │
│  ┌──────────────┐      ┌──────────────┐ │
│  │   GPU 0      │      │   GPU 1      │ │
│  │  Layers 0-9  │◄────►│ Layers 10-19 │ │
│  └──────────────┘      └──────────────┘ │
│         ↓                    ↓           │
└─────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│            Data Parallel Group 2         │
│  ┌──────────────┐      ┌──────────────┐ │
│  │   GPU 2      │      │   GPU 3      │ │
│  │  Layers 0-9  │◄────►│ Layers 10-19 │ │
│  └──────────────┘      └──────────────┘ │
│         ↓                    ↓           │
└─────────────────────────────────────────┘

数据并行度 = world_size / pipeline_stages = 4 / 2 = 2
```

### 6.3 激活检查点

```python
# train.py 第 549-565 行：激活检查点配置
activation_checkpointing = config['activation_checkpointing']

if activation_checkpointing:
    if activation_checkpointing == True:
        # PyTorch 原生检查点
        checkpoint_func = partial(
            torch.utils.checkpoint.checkpoint,
            use_reentrant=config['reentrant_activation_checkpointing']
        )
    elif activation_checkpointing == 'unsloth':
        # Unsloth 优化检查点 (更低内存)
        checkpoint_func = unsloth_checkpoint

    additional_pipeline_module_kwargs.update({
        'activation_checkpoint_interval': 1,           # 每层都检查点
        'checkpointable_layers': model.checkpointable_layers,  # 可检查点的层
        'activation_checkpoint_func': checkpoint_func, # 检查点函数
    })
```

### 6.4 梯度累积

```python
# train.py 第 161-167 行：数据迭代器预取
def get_data_iterator_for_step(dataloader, engine, num_micro_batches=None):
    num_micro_batches = num_micro_batches or engine.micro_batches

    # 非首尾阶段返回空迭代器
    if not (engine.is_first_stage() or engine.is_last_stage()):
        return None

    # 预取所有微批次
    dataloader_iter = iter(dataloader)
    items = [next(dataloader_iter) for _ in range(num_micro_batches)]
    return iter(items)
```

## 7. 优化器系统

### 7.1 优化器类型对比

| 优化器 | 文件 | 特性 | 适用场景 |
|--------|------|------|----------|
| **AdamW8BitKahan** | adamw_8bit.py | 8-bit 量化 + Kahan 求和 | 显存受限场景 |
| **GenericOptim** | generic_optim.py | SM/Muon 优化 | 大规模模型 |
| **Automagic** | automagic.py | 自动优化 | 实验性 |
| **GradientRelease** | gradient_release.py | 梯度释放 | 超大模型 (内存优化) |

### 7.2 GenericOptim 架构

```python
# optimizers/generic_optim.py 核心实现
class GenericOptim(torch.optim.Optimizer):
    def __init__(self, params, second_moment_type='adam'):
        """
        second_moment_type:
        - 'adam': AdamW 风格
        - 'sm': 子空间动量 (SVD/Top-K/均匀/SRHT)
        - 'muon': Muon 优化器
        """
        # 1. 参数分组 (2D vs 非 2D)
        self.param_groups = self._group_params(params)

        # 2. 初始化状态
        for pg in self.param_groups:
            for p in pg['params']:
                state = self.state[p]
                state['step'] = 0
                # AdamW 状态
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
                # SM 状态
                if second_moment_type == 'sm':
                    state['subspace'] = self._init_subspace(p)

    def step(self, closure=None):
        # 1. 获取梯度范数
        grad_norm = self._get_grad_norm()

        # 2. 参数更新
        for pg in self.param_groups:
            for p in pg['params']:
                if p.grad is None:
                    continue

                # AdamW 更新
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = pg['betas']
                step = state['step']

                # 偏置校正
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                # SM 投影 (可选)
                if self.second_moment_type == 'sm':
                    p.data = self._subspace_project(p.data, state['subspace'])

                # 更新步骤
                p.data.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(eps), value=-lr)

                # 权重衰减
                p.data.add_(p.data, alpha=-weight_decay * lr)
```

### 7.3 梯度释放优化器

```python
# optimizers/gradient_release.py 核心思想
class GradientReleaseOptimizerWrapper:
    """
    梯度释放：每 micro-batch 后立即释放梯度
    节省内存，但需要特殊处理动量等
    """
    def __init__(self, optimizers):
        self.optimizers = optimizers

    def step(self):
        # 每个优化器独立 step
        for opt in self.optimizers:
            opt.step()

        # 立即释放梯度 (避免累积)
        for opt in self.optimizers:
            for group in opt.param_groups:
                for p in group['params']:
                    p.grad = None
```

### 7.4 优化器参数组

```python
# models/sdxl.py 示例：分离学习率
def get_param_groups(self, parameters):
    """SDXL 支持 UNet 和文本编码器分离学习率"""
    param_groups = []

    # UNet 参数
    unet_params = []
    text_encoder_1_params = []
    text_encoder_2_params = []

    for name, p in self.diffusion_model.named_parameters():
        if 'model.diffusion_model' in name:
            unet_params.append(p)
        elif 'conditioner.embedders.0.transformer' in name:
            text_encoder_1_params.append(p)
        elif 'conditioner.embedders.1.transformer' in name:
            text_encoder_2_params.append(p)

    param_groups.append({'params': unet_params})
    param_groups.append({'params': text_encoder_1_params})
    param_groups.append({'params': text_encoder_2_params})

    return param_groups
```

## 8. 内存优化技术

### 8.1 块交换 (Block Swap)

```python
# utils/offloading.py 核心思想
class BlockSwapManager:
    def __init__(self, model, blocks_to_swap):
        self.model = model
        self.blocks_to_swap = blocks_to_swap
        self.cpu_blocks = []
        self.gpu_blocks = []

    def enable(self):
        """启用块交换"""
        # 1. 识别可交换的块
        self.identify_swappable_blocks()

        # 2. 将部分块移动到 CPU
        self.offload_blocks_to_cpu()

    def offload_blocks_to_cpu(self):
        """卸载块到 CPU"""
        for block in self.cpu_blocks:
            block.data = block.data.cpu()
            del block.grad  # 释放梯度

    def prepare_block_swap_training(self):
        """准备训练时的块交换"""
        # 将当前需要的块加载到 GPU
        for block in self.gpu_blocks:
            block.data = block.data.cuda()
```

### 8.2 FP8 训练

```python
# models/hunyuan_video.py 示例
model_config = {
    'dtype': 'bfloat16',
    'transformer_dtype': 'float8_e4m3',  # FP8 精度
    'diffusion_model_dtype': 'float8_e4m3',
}
```

### 8.3 内存优化组合

| 技术 | 显存节省 | 性能影响 | 兼容性 |
|------|----------|----------|--------|
| 激活检查点 | 30-50% | 10-20% 慢 | 所有模型 |
| 块交换 | 40-60% | 20-40% 慢 | Wan/Flux/HunyuanVideo/Chroma |
| FP8 训练 | 50% | 轻微 | HunyuanVideo/Cosmos/Wan |
| 梯度释放 | 20-30% | 5-10% 慢 | 所有模型 |

## 9. 评估系统

### 9.1 评估流程

```python
# train.py 第 170-236 行：评估实现
def evaluate(model, model_engine, eval_dataloaders, tb_writer, step):
    """评估模型"""
    model.prepare_block_swap_inference(disable_block_swap=True)

    with torch.no_grad(), isolate_rng():
        # 评估每个数据集
        for name, eval_dataloader in eval_dataloaders.items():
            losses = []
            # 对每个时间步分位数评估
            for quantile in TIMESTEP_QUANTILES_FOR_EVAL:
                loss = evaluate_single(model_engine, eval_dataloader, quantile)
                losses.append(loss)

                # 记录指标
                tb_writer.add_scalar(f'{name}/loss_quantile_{quantile:.2f}', loss, step)
                if wandb_enable:
                    wandb.log({f'{name}/loss_quantile_{quantile:.2f}': loss})

            # 计算平均损失
            avg_loss = sum(losses) / len(losses)
            tb_writer.add_scalar(f'{name}/loss', avg_loss, step)
```

### 9.2 时间步分位数

```python
# train.py 第 39 行
TIMESTEP_QUANTILES_FOR_EVAL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
```

**为什么使用分位数？**
- 扩散模型在不同时间步损失差异很大
- 低分位数 (早期时间步) 损失更稳定
- 高分位数 (后期时间步) 更接近生成质量
- 平均所有分位数提供全面评估

## 10. 支持的模型

### 10.1 完整模型支持矩阵

| #   | 模型 | LoRA | 全参微调 | FP8 | 类型 |
|-----|------|------|----------|-----|------|
| 1   | SDXL | ✅ | ✅ | ❌ | 图像 |
| 2   | Flux | ✅ | ✅ | ✅ | 图像 |
| 3   | LTX-Video | ✅ | ❌ | ❌ | 视频 |
| 4   | HunyuanVideo | ✅ | ❌ | ✅ | 视频 |
| 5   | Cosmos | ✅ | ❌ | ❌ | 视频 |
| 6   | Lumina Image 2.0 | ✅ | ✅ | ❌ | 图像 |
| 7   | Wan2.1 | ✅ | ✅ | ✅ | 图像/视频 |
| 8   | Chroma | ✅ | ✅ | ✅ | 图像 |
| 9   | HiDream | ✅ | ❌ | ✅ | 图像 |
| 10  | SD3 | ✅ | ❌ | ✅ | 图像 |
| 11  | Cosmos-Predict2 | ✅ | ✅ | ✅ | 视频 |
| 12  | OmniGen2 | ✅ | ❌ | ❌ | 图像 |
| 13  | Flux Kontext | ✅ | ✅ | ✅ | 图像 |
| 14  | Wan2.2 | ✅ | ✅ | ✅ | 图像/视频 |
| 15  | Qwen-Image | ✅ | ✅ | ✅ | 图像 |
| 16  | Qwen-Image-Edit | ✅ | ✅ | ✅ | 图像 |
| 17  | HunyuanImage-2.1 | ✅ | ✅ | ✅ | 图像 |
| 18  | AuraFlow | ✅ | ❌ | ✅ | 图像 |
| 19  | Z-Image | ✅ | ✅ | ✅ | 图像 |
| 20  | HunyuanVideo-1.5 | ✅ | ✅ | ✅ | 视频 |

### 10.2 模型分组

**视频模型** (7 个):
- LTX-Video
- HunyuanVideo / HunyuanVideo-1.5
- Cosmos / Cosmos-Predict2
- Wan2.1 / Wan2.2 (t2v 和 i2v)

**图像模型** (13 个):
- SDXL
- Flux 系列 (Flux, Flux Kontext)
- SD3
- Lumina 2.0
- HiDream
- Qwen-Image 系列 (Qwen-Image, Qwen-Image-Edit)
- HunyuanImage-2.1
- AuraFlow
- Z-Image
- Chroma
- OmniGen2

### 10.3 模型特性对比

| 模型 | VAE 类型 | 文本编码器 | 特殊功能 |
|------|----------|-----------|----------|
| SDXL | SDXL-VAE | CLIP-L + CLIP-G | 分离学习率 |
| Flux |ae | T5 | 双块结构 |
| HunyuanVideo | HunyuanVideo-VAE | LLaVA | 视频 + 图像 |
| Wan2.1 | VAE 2.1 | T5 + CLIP | t2v + i2v |
| Qwen-Image | Qwen-VL | Qwen2-VL | 视觉语言 |

## 11. 外部依赖

### 11.1 子模块结构

```
submodules/
├── ComfyUI/                    # 通用推理后端
│   ├── comfy/                 # 核心代码
│   ├── models/                # 模型文件
│   └── extras/                # 扩展节点
├── HunyuanVideo/              # 混元视频 (腾讯)
├── Cosmos/                    # NVIDIA Cosmos
├── Lumina_2/                  # Lumina 图像 (昆仑万维)
├── flow/                      # Flow 匹配模型
├── HiDream/                   # HiDream 图像
├── LTX_Video/                 # LTX 视频 (Lightricks)
├── OmniGen2/                  # OmniGen2 图像
└── HunyuanImage-2.1/          # 混元图像
```

### 11.2 依赖安装

```bash
# 克隆时初始化子模块
git clone --recurse-submodules https://github.com/tdrussell/diffusion-pipe

# 或后续初始化
git submodule init
git submodule update

# 更新依赖
git pull
git submodule update
```

### 11.3 核心 Python 依赖

```
deepspeed==0.17.0       # 分布式训练
diffusers==0.36.0       # 扩散模型
transformers            # 文本编码器
datasets                # 数据集缓存
peft                    # LoRA/IA3
bitsandbytes            # 8-bit 优化器
flash-attn              # 注意力加速 (可选)
imageio[ffmpeg]         # 视频处理
av                      # 视频编解码
safetensors             # 安全权重加载
torchvision             # 图像处理
```

## 12. 配置系统

### 12.1 主配置结构

```toml
# examples/main_example.toml

[output]
output_dir = '/data/training_runs'
dataset = 'examples/dataset.toml'

[training]
epochs = 1000
micro_batch_size_per_gpu = 1
pipeline_stages = 1
gradient_accumulation_steps = 1
gradient_clipping = 1.0
warmup_steps = 100

[model]
type = 'flux'
vae = '/path/to/vae.safetensors'
text_encoders = [
    { type = 't5', path = '/path/to/t5' },
]
diffusion_model = '/path/to/diffusion_model.safetensors'
dtype = 'bfloat16'
guidance = 1.0

[optimizer]
type = 'adamw8bitkahan'
lr = 5e-5
betas = [0.9, 0.99]
weight_decay = 0.01

[adapter]
type = 'lora'
rank = 16
alpha = 16
dropout = 0.0
dtype = 'bfloat16'

[eval]
eval_datasets = [
    { name = 'eval1', config = 'path/to/eval_dataset.toml' }
]
eval_every_n_epochs = 1
eval_before_first_step = true
eval_micro_batch_size_per_gpu = 1

[save]
save_every_n_epochs = 2
checkpoint_every_n_minutes = 120
save_dtype = 'bfloat16'

[optimization]
activation_checkpointing = true
blocks_to_swap = 20
compile = false
```

### 12.2 数据集配置

```toml
# examples/dataset.toml

[[datasets]]
root = '/data/train_images'
caption_prefix = ''
shuffle_capitalization = false
shuffle_delimiter = ', '
shuffle_count = 0

# 视频数据集示例
[[datasets]]
root = '/data/train_videos'
support_video = true
framerate = 16
video_clip_mode = 'single_beginning'
caption_prefix = ''
```

### 12.3 配置验证

```python
# train.py 第 92-137 行
def set_config_defaults(config):
    # 1. 强制设置保存频率
    assert 'save_every_n_epochs' in config or \
           'save_every_n_steps' in config or \
           'save_every_n_examples' in config

    # 2. 数据类型转换
    if 'save_dtype' in config:
        config['save_dtype'] = DTYPE_MAP[config['save_dtype']]

    model_dtype_str = model_config['dtype']
    model_config['dtype'] = DTYPE_MAP[model_dtype_str]

    # 3. LoRA 配置验证
    if 'adapter' in config:
        adapter_config = config['adapter']
        if adapter_config['type'] == 'lora':
            if 'alpha' in adapter_config:
                raise NotImplementedError('alpha must equal rank')
            adapter_config['alpha'] = adapter_config['rank']
```

## 13. 检查点与恢复

### 13.1 检查点类型

```
output_dir/
├── 20250212_07-06-40/           # 训练运行目录
│   ├── config.toml              # 训练配置备份
│   ├── dataset.toml             # 数据集配置备份
│   │
│   ├── epoch1/                  # 保存的模型 (LoRA/完整)
│   │   ├── adapter_model.safetensors
│   │   ├── adapter_config.json
│   │   └── config.json
│   │
│   ├── epoch2/
│   │   └── ...
│   │
│   ├── global_step1234/         # DeepSpeed 检查点
│   │   ├── mp_rank_00_model_states.pt
│   │   ├── optimizer_states.pt
│   │   ├── lr_scheduler.pt
│   │   └── random_states.pt
│   │
│   ├── global_step2468/
│   │   └── ...
│   │
│   └── tensorboard_logs/        # TensorBoard 日志
│       └── events.out.tfevents.*
```

### 13.2 保存流程

```python
# utils/saver.py 核心逻辑
class Saver:
    def __init__(self, config, run_dir, model, dataloader, model_engine):
        self.config = config
        self.run_dir = run_dir
        self.model = model
        self.dataloader = dataloader
        self.model_engine = model_engine

    def save_model(self, save_dir):
        """保存 LoRA 或完整模型"""
        if self.is_adapter:
            # 保存 LoRA
            lora_sd = self.model.lora_model.state_dict()
            self.model.save_adapter(save_dir, lora_sd)
        else:
            # 保存完整模型
            full_sd = self.model.diffusion_model.state_dict()
            self.model.save_model(save_dir, full_sd)

    def save_checkpoint(self, step, examples):
        """保存训练状态检查点"""
        checkpoint_dir = os.path.join(self.run_dir, f'global_step{step}')
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.model_engine.save_checkpoint(
            checkpoint_dir,
            client_state={
                'step': step,
                'examples': examples,
                'custom_loader': self.dataloader.state_dict()
            }
        )
```

### 13.3 恢复流程

```python
# train.py 第 789-812 行
if resume_from_checkpoint:
    # 加载检查点
    load_path, client_state = model_engine.load_checkpoint(
        run_dir,
        load_module_strict=False,
        load_lr_scheduler_states=True,
        load_optimizer_states=True,
    )

    # 恢复数据加载器状态
    if not args.reset_dataloader:
        train_dataloader.load_state_dict(client_state['custom_loader'])

    # 恢复训练状态
    step = client_state['step'] + 1
    examples = client_state.get('examples', step * global_batch_size)
```

## 14. 扩展指南

### 14.1 添加新模型

**步骤 1**: 在 `models/` 创建新文件，例如 `my_model.py`

```python
from models.base import BasePipeline

class MyModelPipeline(BasePipeline):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model_config = config['model']

    def load_diffusion_model(self):
        # 1. 加载 VAE
        self.vae = load_vae(self.model_config['vae'])

        # 2. 加载文本编码器
        self.text_encoders = [
            load_text_encoder(self.model_config['text_encoder'])
        ]

        # 3. 加载扩散模型
        self.transformer = load_transformer(self.model_config['diffusion_model'])

    def get_vae(self):
        return self.vae

    def get_text_encoders(self):
        return self.text_encoders

    def prepare_inputs(self, inputs, timestep_quantile=None):
        # 准备模型输入
        latents = inputs['latents']
        text_embeds = inputs['text_embeds']
        timesteps = self.sample_timesteps(latents.shape[0], timestep_quantile)
        return latents, text_embeds, timesteps

    def to_layers(self):
        # 将模型转换为层列表供 DeepSpeed 分割
        return [
            self.transformer.input_block,
            self.transformer.transformer_blocks,
            self.transformer.out_block,
        ]

    def get_loss_fn(self):
        # 使用默认 MSE 损失或自定义
        return super().get_loss_fn()
```

**步骤 2**: 在 `train.py` 添加映射

```python
elif model_type == 'my_model':
    from models import my_model
    model = my_model.MyModelPipeline(config)
```

**步骤 3**: 更新文档 `docs/supported_models.md`

### 14.2 添加新优化器

**步骤 1**: 在 `optimizers/` 创建文件

```python
import torch

class MyCustomOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, **kwargs):
        defaults = dict(lr=lr, **kwargs)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # 自定义更新逻辑
                p.data.add_(p.grad, alpha=-group['lr'])

        return loss
```

**步骤 2**: 在 `train.py` 添加工厂逻辑

```python
elif optim_type_lower == 'mycustom':
    from optimizers import my_custom_optim
    klass = my_custom_optim.MyCustomOptimizer
```

### 14.3 配置验证扩展

```python
# 在模型类中
def model_specific_dataset_config_validation(self, dataset_config):
    # 验证模型特定配置
    if self.model_config['type'] == 'my_model':
        assert 'my_specific_param' in dataset_config
```

## 15. 性能调优

### 15.1 显存优化优先级

```
高优先级 (影响大):
1. blocks_to_swap = 20-32       # 块交换 (40-60% 节省)
2. activation_checkpointing      # 激活检查点 (30-50% 节省)
3. FP8 训练 (如果支持)           # FP8 量化 (50% 节省)

中优先级 (影响中等):
4. gradient_release              # 梯度释放 (20-30% 节省)
5. 减小 micro_batch_size         # 减小批量 (线性节省)

低优先级 (影响小):
6. compile = true                # torch.compile (5-10% 加速)
```

### 15.2 推荐配置

**24GB VRAM (单卡)**:
```toml
[optimizer]
type = 'AdamW8BitKahan'
lr = 5e-5

[optimization]
activation_checkpointing = true
blocks_to_swap = 20
micro_batch_size_per_gpu = 1
```

**48GB VRAM (单卡/双卡)**:
```toml
[optimizer]
type = 'adamw'
lr = 4e-5

[optimization]
activation_checkpointing = false
blocks_to_swap = 0
micro_batch_size_per_gpu = 2
```

**80GB+ VRAM (多卡)**:
```toml
[optimizer]
type = 'adamw'
lr = 4e-5

[optimization]
activation_checkpointing = false
pipeline_stages = 2
micro_batch_size_per_gpu = 4
```

### 15.3 批量大小建议

| GPU 显存 | micro_batch_size_per_gpu | pipeline_stages | 总批量大小 |
|----------|---------------------------|-----------------|------------|
| 24GB | 1 | 1 | 1 |
| 48GB | 2 | 1 | 2 |
| 48GB | 1 | 2 | 2 |
| 80GB | 4 | 1 | 4 |
| 80GB | 2 | 2 | 4 |
| 4×40GB | 2 | 2 | 8 |

## 16. 故障排查

### 16.1 常见错误

**CUDA OOM**:
```toml
# 增加以下配置
[optimization]
activation_checkpointing = true
blocks_to_swap = 20
micro_batch_size_per_gpu = 1

# 使用 8-bit 优化器
[optimizer]
type = 'AdamW8BitKahan'
```

**NCCL 错误**:
```bash
# 设置环境变量
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
deepspeed --num_gpus=4 train.py --config config.toml
```

**模型加载失败**:
```python
# 检查权重格式和路径
import safetensors
state_dict = safetensors.torch.load_file('/path/to/model.safetensors')
print(state_dict.keys())
```

### 16.2 调试技巧

**查看模型参数**:
```python
# train.py 第 144-155 行
def print_model_info(model):
    print(model)
    for name, module in model.named_modules():
        print(f'{type(module)}: {name}')
        for pname, p in module.named_parameters(recurse=False):
            print(f'  {pname}: {p.numel()} params, {p.dtype}, {p.device}')
```

**测试缓存**:
```bash
# 仅缓存数据，不训练
deepspeed train.py --config config.toml --cache_only

# 强制重新生成缓存
deepspeed train.py --config config.toml --regenerate_cache
```

**调试评估**:
```toml
[eval]
eval_before_first_step = true  # 训练前评估
eval_every_n_steps = 10        # 频繁评估
```

## 17. 最佳实践

### 17.1 训练流程建议

1. **从示例开始**: 复制 `examples/main_example.toml` 并修改
2. **小批量测试**: 先用 micro_batch_size=1 验证
3. **检查损失**: 监控训练损失是否下降
4. **评估验证**: 使用评估集检查泛化能力
5. **保存检查点**: 定期保存避免丢失训练进度
6. **渐进式调参**: 逐渐增加批量大小或学习率

### 17.2 数据集准备

```
数据集结构:
/data/train/
├── images/
│   ├── 001.jpg + 001.txt
│   ├── 002.jpg + 002.txt
│   └── ...
└── videos/
    ├── 001.mp4 + 001.txt
    └── ...

最佳实践:
- 统一图像尺寸 (减少填充)
- 平衡数据分布
- 清理损坏文件
- 高质量标注
```

### 17.3 超参数调优

| 参数 | 建议范围 | 说明 |
|------|----------|------|
| 学习率 | 1e-5 - 5e-5 | LoRA 常用 2e-5 - 5e-5 |
| LoRA 秩 | 16 - 128 | 秩越高容量越大但过拟合 |
| Dropout | 0 - 0.1 | 防止过拟合 |
| 梯度累积 | 1 - 8 | 增加有效批量 |
| Warmup 步数 | 100 - 1000 | 稳定训练初期 |

## 18. 总结

**diffusion-pipe** 是一个生产级别的大规模扩散模型训练框架，具有以下特点：

### 18.1 核心优势

1. **统一性**: 19+ 模型使用同一套训练代码
2. **可扩展性**: 清晰的模块化，易于添加新模型
3. **高效性**: DeepSpeed 管道并行 + 智能缓存
4. **灵活性**: 支持 LoRA、全参微调、混合精度
5. **生产就绪**: 完整的检查点、评估、日志系统

### 18.2 技术亮点

- 基于 DeepSpeed 的混合并行 (数据 + 管道)
- SQLite + 分片文件的高效缓存系统
- 智能的块交换和激活检查点
- 支持 FP8/bf16 等多种精度
- 统一的数据处理管道 (图像 + 视频)

### 18.3 适用场景

- **研究**: 快速实验新模型和数据
- **工业**: 大规模模型训练和微调
- **教育**: 学习扩散模型训练技术
- **开源**: 社区模型训练和分享

### 18.4 未来展望

- 更多模型架构支持 (Sora、Gen-3 等)
- 强化学习优化 (RLHF)
- 多模态融合 (文本、图像、视频、音频)
- 自动超参数优化
- 云原生部署支持

---

**文档版本**: v1.0
**最后更新**: 2025-12-31
**项目地址**: https://github.com/tdrussell/diffusion-pipe

# ODEO AI Studio 开发日志与问题复盘

## 项目开发时间线

2026年1月27日 - 1月30日

---

## 一、主要功能迭代

### Phase 1: 基础框架搭建
- 基于 diffusion-pipe 项目创建 Web API
- 实现 Flask 后端服务
- 创建训练任务管理 API

### Phase 2: 前端界面开发
- 参考火山引擎 Lumi 控制台设计
- 实现暗色主题 UI
- 创建单页应用 (SPA) 架构

### Phase 3: 训练功能完善
- 集成 DeepSpeed 分布式训练
- 实现 GPU 状态监控
- 添加训练进度和 Loss 曲线

### Phase 4: 推理功能集成
- 下载安装 ComfyUI-WanVideoWrapper
- 实现图库管理系统
- 创建推理任务 API

---

## 二、遇到的问题与解决方案

### 1. 依赖安装问题

#### 问题 1.1: Flask 安装失败
```
ERROR: Could not find a version that satisfies the requirement Flask==3.0.0
```
**解决**: 使用国内镜像源
```bash
pip install flask -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 问题 1.2: DeepSpeed 安装超时
```
Name or service not known
```
**解决**: 切换阿里云镜像
```bash
pip install deepspeed -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

#### 问题 1.3: 缺少依赖模块
依次缺少: `pillow`, `toml`, `wandb`, `tensorboard`, `multiprocess`, `torchvision`, `einops`, `diffusers`, `accelerate`

**解决**: 逐个安装
```bash
pip install <package> -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

### 2. 模块导入问题

#### 问题 2.1: ModuleNotFoundError: No module named 'utils.common'
**原因**: `sys.path` 配置不正确，train.py 无法找到项目根目录的 utils 模块

**解决**: 修改 `models/base.py` 中的 `sys.path.insert`
```python
# 修复前
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../submodules/ComfyUI'))

# 修复后
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
```

#### 问题 2.2: ModuleNotFoundError: No module named 'models.gpu'
**原因**: API 内部模块导入使用了绝对路径

**解决**: 改为相对导入
```python
# 修复前
from models.gpu import GPUInfo

# 修复后
from .models.gpu import GPUInfo
```

---

### 3. GPU 相关问题

#### 问题 3.1: GPU 列表从 8 个变成 2 个
**原因**: `pynvml` 初始化不稳定，有时返回错误数据

**解决**: 完全弃用 `pynvml`，改用 `nvidia-smi` 命令
```python
result = subprocess.run(
    ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu',
     '--format=csv,noheader,nounits'],
    capture_output=True, text=True, check=True, timeout=10
)
```

#### 问题 3.2: Invalid device id 推理错误
**原因**: 设置 `CUDA_VISIBLE_DEVICES` 后，torch 只能看到映射后的设备 (从 0 开始)

**解决**: 推理时始终使用 device 0
```python
device_idx = 0  # CUDA_VISIBLE_DEVICES 映射后始终使用 0
torch.cuda.set_device(device_idx)
```

#### 问题 3.3: DeepSpeed 默认使用 GPU0
**原因**: `--num_gpus=1` 会忽略 `CUDA_VISIBLE_DEVICES`

**解决**: 使用 `--include` 参数指定 GPU
```python
cmd = [
    'deepspeed',
    f'--include=localhost:{gpu_id}',
    '--master_port', str(master_port),
    'train.py',
    ...
]
```

---

### 4. 训练相关问题

#### 问题 4.1: 训练配置缺少保存参数
```
AssertionError: 必须设置 save_every_n_epochs 或 save_every_n_steps
```
**解决**: 在生成的 TOML 配置中添加默认保存配置
```python
full_config['save'] = {
    'save_every_n_epochs': 2
}
```

#### 问题 4.2: 端口冲突
```
EADDRINUSE: address already in use, port: 29500
```
**解决**: 动态查找可用端口
```python
def _find_available_port(start=29500, end=29600):
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
    return start
```

#### 问题 4.3: 小数据集 IndexError
```
IndexError: Index 2 out of range for dataset of size 2.
```
**原因**: `num_proc` 超过数据集大小

**解决**: 动态调整进程数
```python
NUM_PROC = min(os.cpu_count() or 8, dataset_size)
```

---

### 5. 模型文件问题

#### 问题 5.1: safetensors 文件损坏
```
SafetensorError: Error while deserializing header: incomplete metadata
```
**解决**: 重新下载损坏的模型文件
```bash
# 使用 hf-mirror.com 镜像
wget https://hf-mirror.com/Wan-AI/Wan2.1-I2V-14B-480P/resolve/main/diffusion_pytorch_model-00004-of-00007.safetensors
```

#### 问题 5.2: 下载速度过慢
**解决**: 使用 aria2c 多线程下载
```bash
aria2c -x 8 -s 8 --allow-overwrite=true <url>
```

---

### 6. 前端问题

#### 问题 6.1: 页面刷新后状态丢失
**解决**: 使用 localStorage 持久化页面状态
```javascript
localStorage.setItem('currentPage', page);
const savedPage = localStorage.getItem('currentPage');
```

#### 问题 6.2: 弹窗不显示
**原因**: CSS 中缺少 `.modal-overlay` 样式

**解决**: 添加必要的 CSS
```css
.modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.6);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
}
```

#### 问题 6.3: 浏览器缓存导致更新不生效
**解决**: 在 CSS/JS 链接添加版本号
```html
<link rel="stylesheet" href="/static/css/main.css?v=19">
<script src="/static/js/app.js?v=19"></script>
```

---

### 7. WanVideoWrapper 下载问题

#### 问题 7.1: GitHub 克隆失败
**原因**: 网络连接不稳定

**解决**: 使用 codeload.github.com 下载 zip
```bash
wget "https://codeload.github.com/kijai/ComfyUI-WanVideoWrapper/zip/refs/heads/main" -O wan.zip
```

---

## 三、性能优化

### 1. GPU 缓存机制
实现 2 秒缓存，避免频繁调用 `nvidia-smi`

### 2. 任务持久化
将任务数据保存到 `tasks.json`，服务重启后自动加载

### 3. 异步任务处理
训练和推理任务在后台线程执行，API 立即返回

---

## 四、未完成/待优化

1. **真正的 ComfyUI 推理集成** - 目前使用模拟推理
2. **多 LoRA 融合** - 已预留接口，未实现
3. **任务队列** - 未实现多任务排队
4. **用户认证** - 无权限控制
5. **日志系统** - 使用 print，未接入专业日志框架

---

## 五、经验总结

1. **依赖管理**: 优先使用国内镜像源
2. **GPU 操作**: 优先使用 subprocess 调用 nvidia-smi，比 pynvml 更稳定
3. **路径处理**: 使用绝对路径，避免相对路径导致的问题
4. **前端缓存**: 始终在静态资源链接添加版本号
5. **错误处理**: 添加详细的日志输出便于排查问题
6. **进程管理**: DeepSpeed 需要特别注意 GPU 分配和端口管理

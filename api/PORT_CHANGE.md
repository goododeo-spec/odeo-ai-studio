# 端口修改说明

## 修改内容

将 API 后端服务端口从 `5000` 修改为 `8080`。

## 修改的文件

### 1. 核心代码文件
- **run.py**: 修改默认端口从 5000 到 8080
- **app.py**: 修改默认端口从 5000 到 8080

### 2. 构建和部署文件
- **Makefile**: 
  - 更新开发模式端口说明
  - 更新生产模式启动命令

### 3. 文档文件
- **README.md**: 
  - 更新所有示例 URL
  - 更新端口配置表格
  - 更新 Dockerfile EXPOSE 指令
  - 更新 Gunicorn 启动示例

- **QUICKSTART.md**: 
  - 更新所有 curl 示例 URL
  - 更新 Python 客户端示例

### 4. 测试和演示文件
- **test_gpu_api.py**: 更新 API 基础 URL
- **demo.py**: 更新 API 基础 URL

## 使用方式

### 启动服务
```bash
# 使用默认端口 8080
python run.py

# 或通过环境变量指定端口
PORT=9000 python run.py

# 使用 Makefile
make dev
make prod
```

### 测试 API
```bash
# 健康检查
curl http://localhost:8080/health

# 获取 GPU 状态
curl http://localhost:8080/api/v1/gpu/status

# 获取可用 GPU
curl http://localhost:8080/api/v1/gpu/available
```

### Python 客户端
```python
import requests

API_BASE = "http://localhost:8080/api/v1"
response = requests.get(f"{API_BASE}/gpu/status")
```

## 环境变量

可以使用 `PORT` 环境变量覆盖默认端口：
```bash
export PORT=9000
python run.py
```

## 验证修改

运行以下命令验证端口修改：
```bash
# 检查配置文件
grep -r "8080" /root/diffusion-pipe/api/ --include="*.py" --include="*.md" --include="Makefile"

# 启动服务
python run.py

# 测试 API
curl http://localhost:8080/health
```

## 注意事项

1. 确保防火墙允许 8080 端口访问
2. 如果使用反向代理（如 Nginx），需要更新代理配置
3. Docker 部署时需要更新端口映射
4. 环境变量 `PORT` 可以覆盖默认端口设置

## 修改时间

2025-12-31

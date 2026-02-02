# Diffusion-Pipe API Makefile

.PHONY: help install test run clean dev stop

# 默认目标
help:
	@echo "Diffusion-Pipe API"
	@echo "==================="
	@echo ""
	@echo "可用命令:"
	@echo "  install    安装依赖"
	@echo "  dev        开发模式启动 (端口 8080)"
	@echo "  prod       生产模式启动"
	@echo "  test       运行 API 测试"
	@echo "  stop       停止 API 服务"
	@echo "  clean      清理日志文件"
	@echo "  logs       查看实时日志"
	@echo ""

# 安装依赖
install:
	@echo "安装依赖..."
	pip install -r requirements.txt
	@echo "依赖安装完成"

# 开发模式启动
dev:
	@echo "启动开发模式..."
	python run.py

# 生产模式启动
prod:
	@echo "启动生产模式..."
	FLASK_ENV=production PORT=8080 python run.py

# 运行测试
test:
	@echo "运行 API 测试..."
	python test_gpu_api.py

# 停止服务
stop:
	@echo "停止 API 服务..."
	pkill -f "python run.py" || true
	@echo "服务已停止"

# 清理日志
clean:
	@echo "清理日志文件..."
	rm -f api.log
	rm -rf logs/*
	@echo "清理完成"

# 查看实时日志
logs:
	@if [ -f api.log ]; then \
		tail -f api.log; \
	else \
		echo "日志文件不存在，请先启动服务"; \
	fi

# 后台启动
background:
	@echo "后台启动 API 服务..."
	nohup python run.py > api.log 2>&1 &
	@echo "服务已在后台启动 (PID: $$!)"
	@echo "查看日志: make logs"
	@echo "停止服务: make stop"

# 检查服务状态
status:
	@echo "检查服务状态..."
	@ps aux | grep "python run.py" | grep -v grep || echo "服务未运行"

# 安装系统依赖 (Ubuntu/Debian)
install-deps:
	@echo "安装系统依赖..."
	sudo apt-get update
	sudo apt-get install -y python3-dev build-essential
	@echo "系统依赖安装完成"

# 创建必要目录
setup-dirs:
	@echo "创建必要目录..."
	mkdir -p /data/logs
	mkdir -p /data/models
	mkdir -p /data/training_runs
	mkdir -p /data/preprocessed
	@echo "目录创建完成"

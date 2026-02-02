import os
import subprocess
import json
from pathlib import Path
from PIL import Image
import torch
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
from transformers import AutoModelForCausalLM, AutoProcessor
from openai import OpenAI

# 初始化千问客户端
def init_qwen_client():
    """初始化千问大模型客户端"""
    api_key = "sk-cebe1cdb99ed44a69d41f194c25ece92"
    if not api_key:
        print("⚠️  警告: 未找到 DASHSCOPE_API_KEY 环境变量，千问优化功能将被跳过")
        return None

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        return client
    except Exception as e:
        print(f"❌ 千问客户端初始化失败: {e}")
        return None

def optimize_prompt_with_qwen(caption, client):
    """
    使用千问大模型优化提示词，输出适合视频生成的英文提示词

    Args:
        caption: 原始中文提示词
        client: 千问客户端

    Returns:
        str: 优化后的英文提示词
    """
    if not client:
        return caption

    if not caption or caption.strip() == "":
        return caption

    try:
        prompt = f"""请将以下图像提示词优化为英文视频生成提示词，要求：

1. 保持核心内容不变
2. 将涉及image等图像相关字眼的描述都去掉
3. 保持简洁明了，长度在50-100词之间
5. 直接输出英文提示词，不需要任何解释

图像提示词：
{caption}

请输出优化后的英文提示词："""

        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {'role': 'user', 'content': prompt}
            ]
        )

        optimized_caption = completion.choices[0].message.content.strip()

        # 清理输出（移除可能的引号或前缀）
        if optimized_caption.startswith('"') and optimized_caption.endswith('"'):
            optimized_caption = optimized_caption[1:-1]
        if optimized_caption.startswith("提示词："):
            optimized_caption = optimized_caption[4:]

        print(f"  - ✅ 千问优化完成")
        print(f"  - 原始: {caption[:50]}...")
        print(f"  - 优化后: {optimized_caption[:50]}...")

        return optimized_caption

    except Exception as e:
        print(f"  - ❌ 千问优化失败: {e}")
        return caption

def fixed_get_imports(filename):
    """修复transformers动态模块导入问题"""
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    try:
        imports.remove("flash_attn")
    except:
        print(f"No flash_attn import to remove")
        pass
    return imports

def generate_caption_for_image(image, caption_method="extra_mixed", model_name="promptgen_base_v2.0",
                              max_new_tokens=1024, num_beams=4, random_prompt=False):
    """
    对单个图像进行提示词反推的核心函数

    Args:
        image: PIL Image对象
        caption_method: 提示词生成方法 ('tags', 'simple', 'detailed', 'extra', 'mixed', 'extra_mixed', 'analyze')
        model_name: 模型名称
        max_new_tokens: 最大生成token数
        num_beams: beam search数量
        random_prompt: 是否随机生成

    Returns:
        str: 生成的提示词
    """
    # 设置设备精度
    attention = 'sdpa'
    precision = 'fp16'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

    # 选择模型
    hg_model = 'MiaoshouAI/Florence-2-base-PromptGen-v2.0'
    if model_name == 'promptgen_large_v2.0':
        hg_model = 'MiaoshouAI/Florence-2-large-PromptGen-v2.0'

    model_name_short = hg_model.rsplit('/', 1)[-1]
    model_path = f"/mnt/disk0/pretrained_models/{model_name_short}"

    # 如果模型不存在，打印信息（实际使用中可能需要下载）
    if not os.path.exists(model_path):
        print(f"模型路径不存在: {model_path}")
        print(f"请确保已下载模型到: {model_path}")
        print(f"或修改model_path为正确的模型路径")
        return ""

    # 加载模型和处理器
    try:
        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                attn_implementation=attention,
                device_map=device,
                torch_dtype=dtype,
                trust_remote_code=True
            ).to(device)

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"加载模型失败: {e}")
        return ""

    # 根据caption_method设置提示词
    if caption_method == 'tags':
        prompt = "<GENERATE_TAGS>"
    elif caption_method == 'simple':
        prompt = "<CAPTION>"
    elif caption_method == 'detailed':
        prompt = "<DETAILED_CAPTION>"
    elif caption_method == 'extra':
        prompt = "<MORE_DETAILED_CAPTION>"
    elif caption_method == 'mixed':
        prompt = "<MIX_CAPTION>"
    elif caption_method == 'extra_mixed':
        prompt = "<MIX_CAPTION_PLUS>"
    else:
        prompt = "<ANALYZE>"

    # 处理图像并生成提示词
    try:
        inputs = processor(text=prompt, images=image, return_tensors="pt", do_rescale=False).to(dtype).to(device)

        do_sample = True if random_prompt else False

        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            early_stopping=False,
            do_sample=do_sample,
            num_beams=num_beams,
        )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(
            generated_text,
            task=prompt,
            image_size=(image.width, image.height)
        )

        return parsed_answer[prompt]
    except Exception as e:
        print(f"生成提示词失败: {e}")
        return ""

def process_video_directory(video_dir, output_dir, prompt_prefix="", caption_method="extra_mixed", use_qwen_optimize=True, frame_number=0):
    """
    处理视频目录的主函数

    Args:
        video_dir: 输入视频目录路径
        output_dir: 输出目录路径
        prompt_prefix: 提示词前缀
        caption_method: 提示词生成方法
        use_qwen_optimize: 是否使用千问优化提示词
        frame_number: 用于提示词反推的帧号（从0开始）
    """
    # 初始化千问客户端
    qwen_client = None
    if use_qwen_optimize:
        print("🔄 初始化千问客户端...")
        qwen_client = init_qwen_client()
        if qwen_client:
            print("✅ 千问客户端初始化成功")
        else:
            print("⚠️  千问优化功能将被禁用")
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有视频文件
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v']
    video_files = []

    for file in os.listdir(video_dir):
        file_ext = os.path.splitext(file)[1].lower()
        if file_ext in video_extensions:
            video_files.append(os.path.join(video_dir, file))

    if not video_files:
        print(f"在目录 {video_dir} 中未找到视频文件")
        return

    # 按文件名排序
    video_files.sort()

    print(f"找到 {len(video_files)} 个视频文件")

    # 处理每个视频
    for i, video_path in enumerate(video_files, 1):
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        print(f"\n处理视频 {i}/{len(video_files)}: {video_name}")

        # 1. 转换视频格式为16fps的mp4
        output_video_path = os.path.join(output_dir, f"{i}.mp4")
        first_frame_path = os.path.join(output_dir, f"{i}.jpg")

        print(f"  - 使用ffmpeg转换视频为16fps并提取第{frame_number}帧...")

        # 使用ffmpeg命令：
        # -r 16: 设置帧率为16fps
        # -y: 自动覆盖同名文件
        # -i: 输入文件
        # -vf "select=eq(n\,0)": 只选择第0帧（首帧）
        # -vframes 1: 只提取一帧
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-r", "16",  # 设置帧率为16fps
            "-y",  # 自动覆盖同名文件
            output_video_path
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            print(f"  - ✅ 视频转换完成: {os.path.basename(output_video_path)}")
        except subprocess.CalledProcessError:
            print(f"  - ❌ 视频转换失败: {video_name}")
            continue
        except FileNotFoundError:
            print("  - ❌ 错误: 未找到ffmpeg，请确保已安装并配置环境变量")
            break

        # 提取指定帧
        extract_frame_cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vf", f"select=eq(n\\,{frame_number})",
            "-vframes", "1",
            "-y",
            first_frame_path
        ]

        try:
            subprocess.run(extract_frame_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            print(f"  - ✅ 第{frame_number}帧提取完成: {os.path.basename(first_frame_path)}")
        except subprocess.CalledProcessError:
            print(f"  - ❌ 第{frame_number}帧提取失败: {video_name}")
            continue
        except FileNotFoundError:
            print("  - ❌ 错误: 未找到ffmpeg，请确保已安装并配置环境变量")
            break

        # 2. 对指定帧进行提示词反推
        if os.path.exists(first_frame_path):
            print(f"  - 对第{frame_number}帧进行提示词反推...")

            # 读取图像
            image = Image.open(first_frame_path).convert("RGB")

            # 生成提示词
            caption = generate_caption_for_image(
                image,
                caption_method=caption_method,
                max_new_tokens=1024,
                num_beams=4,
                random_prompt=False
            )

            if caption:
                # 3. 使用千问优化提示词
                if qwen_client:
                    print(f"  - 使用千问优化提示词...")
                    optimized_caption = optimize_prompt_with_qwen(caption, qwen_client)
                else:
                    optimized_caption = caption

                # 4. 添加前缀并保存为txt文件
                final_caption = f"{prompt_prefix} {optimized_caption}" if prompt_prefix else optimized_caption
                txt_path = os.path.join(output_dir, f"{i}.txt")

                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(final_caption.strip())

                print(f"  - 保存提示词到: {os.path.basename(txt_path)}")
                print(f"  - 提示词预览: {final_caption[:100]}...")
            else:
                print(f"  - 提示词生成失败")
        else:
            print(f"  - 警告: 第{frame_number}帧文件不存在，跳过提示词生成")

    # 删除所有生成的帧图片
    print(f"\n🗑️  清理帧图片...")
    for i in range(1, len(video_files) + 1):
        first_frame_path = os.path.join(output_dir, f"{i}.jpg")
        if os.path.exists(first_frame_path):
            try:
                os.remove(first_frame_path)
                print(f"  - ✅ 已删除: {os.path.basename(first_frame_path)}")
            except Exception as e:
                print(f"  - ❌ 删除失败 {os.path.basename(first_frame_path)}: {e}")

    print(f"✅ 帧图片清理完成")

def main():
    """主函数"""
    # 读取配置文件
    config_path = os.path.join(os.path.dirname(__file__), "config.json")

    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
        return

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        video_directory = config.get("video_directory", "")
        output_directory = config.get("output_directory", "")
        prompt_prefix = config.get("prompt_prefix", "")
        caption_method = config.get("caption_method", "detailed")
        use_qwen_optimize = config.get("use_qwen_optimize", True)
        frame_number = config.get("frame_number", 0)

        if not video_directory:
            print("错误: 配置文件中未设置 video_directory")
            return

        if not output_directory:
            print("错误: 配置文件中未设置 output_directory")
            return

    except json.JSONDecodeError as e:
        print(f"错误: 配置文件格式不正确: {e}")
        return
    except Exception as e:
        print(f"错误: 读取配置文件失败: {e}")
        return

    if not os.path.exists(video_directory):
        print(f"错误: 视频目录不存在: {video_directory}")
        return

    print(f"\n开始处理视频...")
    print(f"视频目录: {video_directory}")
    print(f"输出目录: {output_directory}")
    print(f"提示词前缀: {prompt_prefix if prompt_prefix else '无'}")
    print(f"提示词方法: {caption_method}")
    print(f"千问优化: {'启用' if use_qwen_optimize else '禁用'}")
    print(f"使用帧号: {frame_number}")
    print("-" * 50)

    try:
        process_video_directory(video_directory, output_directory, prompt_prefix, caption_method, use_qwen_optimize, frame_number)
        print("\n处理完成!")
    except Exception as e:
        print(f"\n处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

# /mnt/disk0/lora数据集/2
# /mnt/disk0/train_data/2
# A trending dance move, arms and legs rapidly crisscrossing in a fast-paced dance.
# The character jumps up, turns body sideways to the camera, bends over, and starts twerking.
# detailed
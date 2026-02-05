#!/usr/bin/env python
"""
ComfyUI Wan2.1 推理脚本 - 使用 API 格式 workflow
直接使用 ComfyUI 导出的 API 格式 JSON，而不是手动解析 workflow

Workflow 节点映射:
- 节点 58 (LoadImage) ← 图库选择的图片
- 节点 71 (WanVideoLoraSelect) ← 选择的 LoRA 模型
- 节点 81 (TextToLowercase) ← 触发词输入
- 节点 30 (VHS_VideoCombine) → 输出结果
"""
import os
import sys
import json
import time
import argparse
import requests
import shutil
import hashlib
from pathlib import Path

# 路径配置
COMFYUI_URL = "http://127.0.0.1:8188"
COMFYUI_INPUT_DIR = Path("/home/disk2/comfyui/input")
COMFYUI_OUTPUT_DIR = Path("/home/disk2/comfyui/output")
COMFYUI_LORAS_DIR = Path("/home/disk2/comfyui/models/loras")
WORKFLOW_PATH = Path("/home/disk2/comfyui/user/default/workflows/wanvideo_2_1_14B_I2V_odeo.json")

# 节点 ID 映射（对应 wanvideo_2_1_14B_I2V_odeo.json）
NODE_LOAD_IMAGE = "58"        # LoadImage - 输入图片
NODE_LORA_SELECT = "71"       # WanVideoLoraSelect - 用户 LoRA
NODE_TRIGGER_WORD = "81"      # TextToLowercase - 触发词
NODE_OUTPUT_VIDEO = "30"      # VHS_VideoCombine - 输出视频


def parse_args():
    parser = argparse.ArgumentParser(description='ComfyUI Wan2.1 Video Inference')
    
    # 必需参数
    parser.add_argument('--lora_path', type=str, required=True, 
                        help='Path to LoRA file (maps to Node 71)')
    parser.add_argument('--trigger_word', type=str, required=True, 
                        help='Trigger word for prompt (maps to Node 81)')
    parser.add_argument('--image_path', type=str, required=True, 
                        help='Input image path (maps to Node 58)')
    parser.add_argument('--output', type=str, required=True, 
                        help='Output video path (from Node 30)')
    
    # 可选参数
    parser.add_argument('--lora_strength', type=float, default=1.0, 
                        help='LoRA strength (0-1)')
    parser.add_argument('--gpu', type=int, default=4, 
                        help='GPU ID (4-7)')
    parser.add_argument('--seed', type=int, default=-1, 
                        help='Random seed (-1 for random)')
    parser.add_argument('--use_auto_caption', action='store_true', default=True,
                        help='Use QwenVL to auto-caption image (default: True)')
    parser.add_argument('--no_auto_caption', action='store_true',
                        help='Disable auto-caption, use trigger word only')
    parser.add_argument('--num_frames', type=int, default=81,
                        help='Number of frames to generate')
    parser.add_argument('--steps', type=int, default=4,
                        help='Number of sampling steps')
    parser.add_argument('--cfg', type=float, default=1.0,
                        help='CFG scale')
    
    return parser.parse_args()


def get_lora_relative_path(lora_path):
    """将 LoRA 绝对路径转换为 ComfyUI loras 目录的相对路径"""
    lora_path = Path(lora_path)
    
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA 文件不存在: {lora_path}")
    
    # 创建一个唯一的符号链接名称
    path_hash = hashlib.md5(str(lora_path).encode()).hexdigest()[:8]
    lora_filename = f"infer_{path_hash}_{lora_path.name}"
    
    dest_path = COMFYUI_LORAS_DIR / lora_filename
    
    # 创建或更新符号链接
    if dest_path.exists() or dest_path.is_symlink():
        if dest_path.is_symlink():
            current_target = os.readlink(dest_path)
            if current_target != str(lora_path):
                os.remove(dest_path)
                os.symlink(str(lora_path), str(dest_path))
    else:
        os.symlink(str(lora_path), str(dest_path))
    
    print(f"[Inference] LoRA 符号链接: {lora_filename}")
    return lora_filename


def prepare_image(image_path):
    """将图片复制到 ComfyUI input 目录"""
    img_src = Path(image_path)
    if not img_src.exists():
        raise FileNotFoundError(f"图片不存在: {image_path}")
    
    img_dest = COMFYUI_INPUT_DIR / img_src.name
    if not img_dest.exists() or img_src.stat().st_mtime > img_dest.stat().st_mtime:
        shutil.copy2(img_src, img_dest)
    
    return img_src.name


def create_api_prompt(lora_name, trigger_word, image_name, lora_strength=1.0, seed=-1, 
                       use_auto_caption=True, num_frames=81, steps=4, cfg=1.0):
    """
    创建完整的 ComfyUI API prompt
    
    Workflow 节点映射:
    - 节点 58 (LoadImage) ← image_name (图库选择的图片)
    - 节点 71 (WanVideoLoraSelect) ← lora_name, lora_strength (选择的 LoRA 模型)
    - 节点 81 (TextToLowercase) ← trigger_word (触发词输入)
    - 节点 30 (VHS_VideoCombine) → 输出结果
    
    Args:
        lora_name: LoRA 文件名（在 ComfyUI loras 目录中）
        trigger_word: 触发词
        image_name: 输入图片文件名（在 ComfyUI input 目录中）
        lora_strength: LoRA 强度 (0-1)
        seed: 随机种子，-1 表示随机
        use_auto_caption: 是否使用 QwenVL 自动描述图片（与触发词拼接）
        num_frames: 生成帧数
        steps: 采样步数
        cfg: CFG scale
    """
    actual_seed = seed if seed > 0 else int(time.time()) % 2147483647
    
    # 基础 API prompt
    api_prompt = {
        # === 文本处理部分 ===
        
        # Node 81: 触发词转小写（对应工作流中的 TextToLowercase）
        "81": {
            "class_type": "TextToLowercase",
            "inputs": {
                "texts": trigger_word
            }
        },
        
        # Node 11: T5 文本编码器
        "11": {
            "class_type": "LoadWanVideoT5TextEncoder",
            "inputs": {
                "model_name": "models_t5_umt5-xxl-enc-bf16.pth",
                "precision": "bf16",
                "load_device": "offload_device",
                "quantization": "disabled"
            }
        },
        
        # === 图像加载部分 ===
        
        # Node 58: 图片加载（对应工作流中的 LoadImage）
        "58": {
            "class_type": "LoadImage",
            "inputs": {
                "image": image_name
            }
        },
        
        # Node 73: 图像缩放
        "73": {
            "class_type": "WanVideoImageResizeToClosest",
            "inputs": {
                "image": ["58", 0],
                "generation_width": 832,
                "generation_height": 480,
                "aspect_ratio_preservation": "keep_input"
            }
        },
        
        # === CLIP Vision 部分 ===
        
        # Node 59: CLIP Vision 加载器
        "59": {
            "class_type": "CLIPVisionLoader",
            "inputs": {
                "clip_name": "open-clip-xlm-roberta-large-vit-huge-14_visual_fp16.safetensors"
            }
        },
        
        # Node 65: CLIP Vision 编码
        "65": {
            "class_type": "WanVideoClipVisionEncode",
            "inputs": {
                "clip_vision": ["59", 0],
                "image_1": ["73", 0],
                "strength_1": 1,
                "strength_2": 1,
                "crop": "center",
                "combine_embeds": "average",
                "force_offload": True,
                "tiles": 0,
                "ratio": 0.2
            }
        },
        
        # === VAE 部分 ===
        
        # Node 38: VAE 加载器
        "38": {
            "class_type": "WanVideoVAELoader",
            "inputs": {
                "model_name": "Wan2_1_VAE_bf16.safetensors",
                "precision": "bf16",
                "use_cpu_cache": False,
                "verbose": False
            }
        },
        
        # Node 63: 图像到视频编码
        "63": {
            "class_type": "WanVideoImageToVideoEncode",
            "inputs": {
                "vae": ["38", 0],
                "clip_embeds": ["65", 0],
                "start_image": ["73", 0],
                "width": ["73", 1],
                "height": ["73", 2],
                "num_frames": num_frames,
                "noise_aug_strength": 0.03,
                "start_latent_strength": 1,
                "end_latent_strength": 1,
                "force_offload": True,
                "fun_or_fl2v_model": False,
                "tiled_vae": False,
                "augment_empty_frames": 0
            }
        },
        
        # === LoRA 部分 ===
        
        # Node 71: 用户 LoRA（对应工作流中的 WanVideoLoraSelect）
        "71": {
            "class_type": "WanVideoLoraSelect",
            "inputs": {
                "lora": lora_name,
                "strength": lora_strength,
                "low_mem_load": False
            }
        },
        
        # Node 69: 第二个 LoRA (distill LoRA for faster inference)
        "69": {
            "class_type": "WanVideoLoraSelect",
            "inputs": {
                "prev_lora": ["71", 0],
                "lora": "Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors",
                "strength": 1,
                "low_mem_load": False
            }
        },
        
        # === 模型部分 ===
        
        # Node 22: 模型加载器
        "22": {
            "class_type": "WanVideoModelLoader",
            "inputs": {
                "lora": ["69", 0],
                "model": "Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors",
                "base_precision": "fp16",
                "quantization": "fp8_e4m3fn",
                "load_device": "offload_device",
                "attention_mode": "sdpa",
                "rms_norm_function": "default"
            }
        },
        
        # Node 39: Block Swap 配置
        "39": {
            "class_type": "WanVideoBlockSwap",
            "inputs": {
                "blocks_to_swap": 10,
                "offload_img_emb": False,
                "offload_txt_emb": False,
                "use_non_blocking": True,
                "vace_blocks_to_swap": 0,
                "prefetch_blocks": 0,
                "block_swap_debug": False
            }
        },
        
        # Node 70: 设置 Block Swap
        "70": {
            "class_type": "WanVideoSetBlockSwap",
            "inputs": {
                "model": ["22", 0],
                "block_swap_args": ["39", 0]
            }
        },
        
        # === 采样部分 ===
        
        # Node 27: 采样器
        "27": {
            "class_type": "WanVideoSampler",
            "inputs": {
                "model": ["70", 0],
                "image_embeds": ["63", 0],
                "text_embeds": ["16", 0],
                "steps": steps,
                "cfg": cfg,
                "shift": 5,
                "seed": actual_seed,
                "scheduler": "dpm++_sde",
                "force_offload": True,
                "riflex_freq_index": 0,
                "denoise_strength": 1,
                "batched_cfg": "",
                "rope_function": "comfy",
                "start_step": 0,
                "end_step": -1,
                "add_noise_to_samples": False
            }
        },
        
        # === 解码和输出部分 ===
        
        # Node 28: 解码器
        "28": {
            "class_type": "WanVideoDecode",
            "inputs": {
                "vae": ["38", 0],
                "samples": ["27", 0],
                "enable_vae_tiling": False,
                "tile_x": 272,
                "tile_y": 272,
                "tile_stride_x": 144,
                "tile_stride_y": 128,
                "normalization": "default"
            }
        },
        
        # Node 72: RIFE 帧插值
        "72": {
            "class_type": "RIFE VFI",
            "inputs": {
                "frames": ["28", 0],
                "ckpt_name": "rife47.pth",
                "clear_cache_after_n_frames": 10,
                "multiplier": 2,
                "fast_mode": True,
                "ensemble": True,
                "scale_factor": 1
            }
        },
        
        # Node 30: 视频输出（对应工作流中的 VHS_VideoCombine）
        "30": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["72", 0],
                "frame_rate": 32,
                "loop_count": 0,
                "filename_prefix": "WanVideoWrapper_I2V",
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 19,
                "save_metadata": True,
                "trim_to_audio": False,
                "pingpong": False,
                "save_output": True
            }
        }
    }
    
    # 根据是否使用自动描述来配置文本编码
    if use_auto_caption:
        # 使用 QwenVL 自动描述 + 触发词拼接
        # Node 77: QwenVL 图片描述
        api_prompt["77"] = {
            "class_type": "AILab_QwenVL",
            "inputs": {
                "image": ["58", 0],
                "model_name": "Qwen3-VL-8B-Instruct",
                "quantization": "None (FP16)",
                "attention_mode": "auto",
                "preset_prompt": "🖼️ Detailed Description",
                "custom_prompt": "你是一名图像反推提示词专家：描述人物的外貌、发型、身材、着装，以及背景。\n- 不要描述人物的姿势、动作\n- 不要描写手拿物品\n输出规则：仅输出英文、单段、≤500 characters，不要任何解释/标题/列表/JSON/前缀，不要有任何描述以外的废话。",
                "max_tokens": 256,
                "keep_model_loaded": True,
                "seed": actual_seed
            }
        }
        
        # Node 79: Prompt 拼接（触发词 + 自动描述）
        api_prompt["79"] = {
            "class_type": "easy promptConcat",
            "inputs": {
                "prompt1": ["81", 0],  # 触发词（小写）
                "prompt2": ["77", 0],  # QwenVL 描述
                "separator": " "
            }
        }
        
        # Node 16: 文本编码 - 使用拼接后的 prompt
        api_prompt["16"] = {
            "class_type": "WanVideoTextEncode",
            "inputs": {
                "t5": ["11", 0],
                "model_to_offload": ["22", 0],
                "positive_prompt": ["79", 0],  # 使用拼接结果
                "negative_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                "force_offload": True,
                "use_disk_cache": False,
                "device": "gpu"
            }
        }
    else:
        # 直接使用触发词（小写）- 不使用节点引用，因为 TextToLowercase 输出是 list
        api_prompt["16"] = {
            "class_type": "WanVideoTextEncode",
            "inputs": {
                "t5": ["11", 0],
                "model_to_offload": ["22", 0],
                "positive_prompt": trigger_word.lower(),  # 直接传入小写的触发词字符串
                "negative_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                "force_offload": True,
                "use_disk_cache": False,
                "device": "gpu"
            }
        }
    
    return api_prompt


def queue_prompt(prompt, client_id=None):
    """向 ComfyUI 提交 prompt"""
    p = {"prompt": prompt}
    if client_id:
        p["client_id"] = client_id
    
    data = json.dumps(p).encode('utf-8')
    response = requests.post(f"{COMFYUI_URL}/prompt", data=data, headers={'Content-Type': 'application/json'})
    
    if response.status_code != 200:
        raise Exception(f"提交任务失败: {response.status_code} - {response.text}")
    
    return response.json()


def get_history(prompt_id):
    """获取任务历史"""
    response = requests.get(f"{COMFYUI_URL}/history/{prompt_id}")
    if response.status_code == 200:
        return response.json()
    return None


def wait_for_completion(prompt_id, timeout=1800):
    """等待任务完成"""
    start_time = time.time()
    last_print = 0
    
    while time.time() - start_time < timeout:
        history = get_history(prompt_id)
        if history and prompt_id in history:
            return history[prompt_id]
        
        elapsed = int(time.time() - start_time)
        if elapsed - last_print >= 30:
            print(f"[Inference] 等待推理完成... ({elapsed}s)")
            last_print = elapsed
        
        time.sleep(2)
    
    raise TimeoutError(f"任务超时 ({timeout}秒)")


def main():
    args = parse_args()
    
    # 设置 CUDA 设备 (注意：ComfyUI 可能有自己的设备管理)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # 确定是否使用自动描述
    use_auto_caption = args.use_auto_caption and not args.no_auto_caption
    
    print(f"[Inference] === ComfyUI Wan2.1 推理 ===")
    print(f"[Inference] Workflow 节点映射:")
    print(f"[Inference]   节点 58 (LoadImage) ← {args.image_path}")
    print(f"[Inference]   节点 71 (WanVideoLoraSelect) ← {args.lora_path}")
    print(f"[Inference]   节点 81 (TextToLowercase) ← {args.trigger_word}")
    print(f"[Inference]   节点 30 (VHS_VideoCombine) → {args.output}")
    print(f"[Inference] ---")
    print(f"[Inference] LoRA 强度: {args.lora_strength}")
    print(f"[Inference] GPU: {args.gpu}")
    print(f"[Inference] 自动描述: {'启用' if use_auto_caption else '禁用'}")
    print(f"[Inference] 帧数: {args.num_frames}, 步数: {args.steps}, CFG: {args.cfg}")
    
    try:
        # 检查 ComfyUI 是否运行
        try:
            response = requests.get(f"{COMFYUI_URL}/system_stats", timeout=5)
            if response.status_code != 200:
                raise Exception("ComfyUI 未运行")
        except Exception as e:
            raise Exception(f"无法连接到 ComfyUI ({COMFYUI_URL}): {e}")
        
        print(f"[Inference] progress: 5%")
        
        # 准备 LoRA（节点 71）
        lora_name = get_lora_relative_path(args.lora_path)
        print(f"[Inference] 节点 71 LoRA 名称: {lora_name}")
        print(f"[Inference] progress: 8%")
        
        # 准备图片（节点 58）
        image_name = prepare_image(args.image_path)
        print(f"[Inference] 节点 58 图片名称: {image_name}")
        print(f"[Inference] progress: 10%")
        
        # 创建 API prompt（包含所有节点配置）
        api_prompt = create_api_prompt(
            lora_name=lora_name,
            trigger_word=args.trigger_word,  # 节点 81
            image_name=image_name,            # 节点 58
            lora_strength=args.lora_strength, # 节点 71 强度
            seed=args.seed,
            use_auto_caption=use_auto_caption,
            num_frames=args.num_frames,
            steps=args.steps,
            cfg=args.cfg
        )
        print(f"[Inference] 已创建 API prompt（节点 81 触发词: {args.trigger_word}）")
        print(f"[Inference] progress: 15%")
        
        # 保存 debug prompt
        debug_path = Path(args.output).parent / "debug_prompt.json"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with open(debug_path, 'w') as f:
            json.dump(api_prompt, f, indent=2)
        print(f"[Inference] 调试文件: {debug_path}")
        
        # 提交任务
        result = queue_prompt(api_prompt)
        prompt_id = result.get('prompt_id')
        if not prompt_id:
            raise Exception(f"提交任务失败: {result}")
        
        print(f"[Inference] 已提交任务: {prompt_id}")
        print(f"[Inference] progress: 20%")
        
        # 等待完成
        history = wait_for_completion(prompt_id)
        print(f"[Inference] progress: 90%")
        
        # 检查是否有错误
        if 'status' in history and history['status'].get('status_str') == 'error':
            error_msg = history['status'].get('messages', [])
            raise Exception(f"推理执行错误: {error_msg}")
        
        # 获取输出
        outputs = history.get('outputs', {})
        
        # 节点 30 是输出节点
        output_node = outputs.get('30', {})
        if not output_node:
            for node_id, node_out in outputs.items():
                if 'videos' in node_out or 'gifs' in node_out:
                    output_node = node_out
                    break
        
        # 获取视频文件
        video_info = None
        if 'videos' in output_node:
            video_info = output_node['videos'][0] if output_node['videos'] else None
        elif 'gifs' in output_node:
            video_info = output_node['gifs'][0] if output_node['gifs'] else None
        
        if video_info:
            output_filename = video_info.get('filename')
            output_subfolder = video_info.get('subfolder', '')
            
            src_path = COMFYUI_OUTPUT_DIR / output_subfolder / output_filename if output_subfolder else COMFYUI_OUTPUT_DIR / output_filename
            
            if src_path.exists():
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, output_path)
                
                print(f"[Inference] progress: 100%")
                print(f"[Inference] ✓ 推理完成: {output_path}")
                print(f"[Inference] 文件大小: {output_path.stat().st_size / 1024:.1f} KB")
                return 0
        
        print(f"[Inference] ✗ 未找到输出视频")
        print(f"[Inference] 输出内容: {outputs}")
        return 1
        
    except Exception as e:
        print(f"[Inference] ✗ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

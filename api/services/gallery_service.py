"""
图库服务 - 管理测试图片
"""
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import json

# 图库根目录
GALLERY_ROOT = Path("/home/disk2/lora_training/gallery")
GALLERY_ROOT.mkdir(parents=True, exist_ok=True)


class GalleryService:
    """图库管理服务"""
    
    def __init__(self):
        self._config_file = GALLERY_ROOT / "gallery_config.json"
        self._load_config()
        print(f"[Gallery] 图库服务初始化完成, 根目录: {GALLERY_ROOT}")
    
    def _load_config(self):
        """加载配置"""
        if self._config_file.exists():
            with open(self._config_file, 'r') as f:
                self._config = json.load(f)
        else:
            self._config = {"folders": []}
            self._save_config()
    
    def _save_config(self):
        """保存配置"""
        with open(self._config_file, 'w') as f:
            json.dump(self._config, f, indent=2, ensure_ascii=False)
    
    def list_folders(self) -> List[Dict[str, Any]]:
        """列出所有文件夹"""
        folders = []
        
        for item in GALLERY_ROOT.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # 统计图片数量
                count = sum(1 for f in item.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp'])
                
                folders.append({
                    'name': item.name,
                    'path': str(item),
                    'count': count,
                    'created_at': datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                })
        
        folders.sort(key=lambda x: x['name'])
        return folders
    
    def create_folder(self, name: str) -> Dict[str, Any]:
        """创建文件夹"""
        folder_path = GALLERY_ROOT / name
        if folder_path.exists():
            raise ValueError(f"文件夹 '{name}' 已存在")
        
        folder_path.mkdir(parents=True, exist_ok=True)
        
        return {
            'name': name,
            'path': str(folder_path),
            'count': 0,
            'created_at': datetime.now().isoformat()
        }
    
    def delete_folder(self, name: str) -> bool:
        """删除文件夹"""
        folder_path = GALLERY_ROOT / name
        if not folder_path.exists():
            return False
        
        shutil.rmtree(folder_path)
        return True
    
    def list_images(self, folder: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出图片"""
        images = []
        
        if folder:
            search_path = GALLERY_ROOT / folder
        else:
            search_path = GALLERY_ROOT
        
        if not search_path.exists():
            return images
        
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
            pattern = f"**/{ext}" if not folder else ext
            for img_path in search_path.glob(pattern):
                if img_path.is_file():
                    rel_path = img_path.relative_to(GALLERY_ROOT)
                    images.append({
                        'id': str(rel_path),
                        'name': img_path.name,
                        'folder': str(img_path.parent.relative_to(GALLERY_ROOT)) if img_path.parent != GALLERY_ROOT else None,
                        'path': str(img_path),
                        'url': f"/api/v1/gallery/image/{rel_path}",
                        'size': img_path.stat().st_size,
                        'created_at': datetime.fromtimestamp(img_path.stat().st_mtime).isoformat()
                    })
        
        images.sort(key=lambda x: x['created_at'], reverse=True)
        return images
    
    def upload_image(self, folder: str, file_content: bytes, filename: str) -> Dict[str, Any]:
        """上传图片"""
        folder_path = GALLERY_ROOT / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        
        # 确保文件名唯一
        base_name = Path(filename).stem
        ext = Path(filename).suffix
        final_name = filename
        counter = 1
        
        while (folder_path / final_name).exists():
            final_name = f"{base_name}_{counter}{ext}"
            counter += 1
        
        file_path = folder_path / final_name
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        rel_path = file_path.relative_to(GALLERY_ROOT)
        return {
            'id': str(rel_path),
            'name': final_name,
            'folder': folder,
            'path': str(file_path),
            'url': f"/api/v1/gallery/image/{rel_path}",
            'size': len(file_content)
        }
    
    def delete_image(self, image_id: str) -> bool:
        """删除图片"""
        image_path = GALLERY_ROOT / image_id
        if not image_path.exists():
            return False
        
        image_path.unlink()
        return True
    
    def get_image_info(self, image_id: str) -> Optional[Dict[str, Any]]:
        """获取图片信息（包含尺寸）"""
        image_path = GALLERY_ROOT / image_id
        if not image_path.exists():
            return None
        
        # 获取图片尺寸
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                width, height = img.size
        except:
            width, height = 0, 0
        
        return {
            'id': image_id,
            'name': image_path.name,
            'path': str(image_path),
            'url': f"/api/v1/gallery/image/{image_id}",
            'width': width,
            'height': height,
            'size': image_path.stat().st_size
        }


# 全局实例
gallery_service = GalleryService()

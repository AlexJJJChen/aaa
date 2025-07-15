# -*- coding: utf-8 -*-
# ======================================================================================
# OMNI-PARSER: UTILITIES MODULE (EXPERT VERIFIED V3)
#
# 包含项目范围内使用的通用辅助函数。
# Contains general-purpose helper functions used across the project.
# This version is a 1:1 match with the helper functions in the original omni_parser.py.
# ======================================================================================
import os
import base64
import io
from tqdm import tqdm
from config import Config

def encode_image_to_base64(image_path):
    """将图像文件编码为base64字符串。"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        tqdm.write(f" [ERROR] 编码失败，图像文件未找到: {image_path}")
        return None
    except Exception as e:
        tqdm.write(f" [ERROR] 编码图像失败 {image_path}: {e}")
        return None

def pil_to_base64(pil_image):
    """将PIL Image对象转换为base64字符串。"""
    import io
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def setup_directories(base_dir):
    """创建所有需要的输出子目录"""
    dirs = Config.get_subdirectories(base_dir)
    for key, path in dirs.items():
        if key.startswith("DIR"):
            os.makedirs(path, exist_ok=True)
    return dirs

def calculate_iou(box1, box2):
    """
    计算两个包围盒（bounding box）的交并比 (Intersection over Union, IoU)。
    """
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0.0
        
    iou = intersection_area / union_area
    return iou

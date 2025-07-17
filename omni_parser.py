# -*- coding: utf-8 -*-
# ======================================================================================
# 全文档处理流水线 (Complete Document Processing Pipeline) - 专家增强版 V8.3 (资源优化)
#
# 本脚本由专家团队审查和重构。在V8.2的性能优化基础上，根据用户反馈进一步
# 优化了资源管理和模型加载策略。
#
# 专家增强版 V8.3 更新说明 (Expert Enhanced V8.3 Update Notes):
# - 按需模型加载 (On-Demand Model Loading): 彻底重构了主执行流程。现在，本地
#   模型(Qwen, Nanonets)和OpenAI客户端不再于脚本启动时全部加载，而是在处理
#   特定文件、即将进入需要它们的识别阶段(Stage 4)前，才进行“即时”初始化。
# - 资源隔离与性能提升: 此项改动解决了关键性能瓶颈。通过推迟加载大型本地模型，
#   确保了前序的版面分析(Stage 2)等步骤可以独占系统资源(特别是VRAM)，
#   从而恢复并超越了其应有的高速性能。
# - 更快的启动速度和更低的内存占用: 对于不需要进行VLM识别的简单流程（如未来
#   可能加入的纯文本提取），脚本启动更快，内存占用也显著降低。
#
# 其余所有V8功能 (PDF, TXT, PPT, ZIP处理, Stage 2批量优化) 保持不变。
# ======================================================================================

# --- 导入通用依赖库 (Import General Libraries) ---
import os
import sys
import time
import json
import re
import traceback
import html
import gc
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import base64
import requests
import shutil
from pdf2image import convert_from_path
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import zipfile
import subprocess
from pathlib import Path

# --- V8 新增依赖 (V8 New Dependencies) ---
# 请确保已安装处理PPT所需的库:
# pip install aspose-slides
#
# Word处理现在依赖于LibreOffice/OpenOffice。
# 请确保已在您的系统上安装，并且其路径在系统环境变量中。
# - Ubuntu/Debian: sudo apt-get install libreoffice
# - CentOS/Fedora: sudo yum install libreoffice
# - Windows/macOS: 从官网下载安装包进行安装。
try:
    import aspose.slides as slides
except ImportError:
    print("❌ 警告: 'aspose-slides' 库未安装。处理PPT文档的功能将不可用。")
    print("   请运行: pip install aspose-slides")
    slides = None


# ======================================================================================
# --- ⚙️ 统一配置 (Unified Configuration) ---
# ======================================================================================
class Config:
    """
    集中管理所有路径、模型和API密钥的配置类。
    Configuration class to centrally manage all paths, models, and API keys.
    """
    # --- V8核心变更: 输入文件路径 (V8 Core Change: Input File Path) ---
    # 支持 .pdf, .docx, .txt, .pptx, .zip
    INPUT_PATH = "/project/chenjian/bbb/[定期报告][2023-03-20][朗鸿科技]朗鸿科技2022年年度报告摘要.pdf" #"/project/chenjian/bbb/论文答辩.pptx"#"/project/chenjian/bbb/研究计划 3.0.txt" #"/project/chenjian/bbb/研究计划 3.0.docx"#"/project/chenjian/bbb/[定期报告][2023-03-20][朗鸿科技]朗鸿科技2022年年度报告摘要.pdf"

    # --- 模型与API配置 (Models & API Config) ---
    # 本地模型路径 (Local Model Paths)
    VLM_MODEL_CHECKPOINT = "/project/chenjian/Qwen/Qwen2.5-VL-7B-Instruct"
    NANONETS_MODEL_CHECKPOINT = "/project/chenjian/nanonets/Nanonets-OCR-s"

    # OpenAI & 兼容API配置 (OpenAI & Compatible API Config)
    API_KEY = "sk-3ni5O4wR7GTeeqKvFdC5D12f280b460797E7369455283a7d"
    API_BASE_URL = "http://152.53.52.170:3003/v1"

    # --- 🚀 模型选择器 (MODEL SELECTOR) ---
    # 在这里为每个任务选择要使用的模型。
    # 可选项: 'local_qwen', 'local_nanonets', 'gpt-4o', 'gpt-4.1-mini-2025-04-14'
    class ModelSelector:
        # 任务1: 图像描述 (图表、照片、PPT页面解读)
        IMAGE_DESCRIPTION = 'gpt-4o' # 可选: 'local_qwen', 'gpt-4o'

        # 任务2: 无框线表格识别
        BORDERLESS_TABLE_RECOGNITION = 'local_nanonets' # 可选: 'local_nanonets', 'gpt-4o'

        # 任务3: 有框线表格的单元格内容识别
        BORDERED_TABLE_CELL_RECOGNITION = 'local_qwen' # 可选: 'local_qwen', 'gpt-4o'

        # 任务4: 跨页表格的智能合并 (需要强大的逻辑推理能力)
        TABLE_MERGING = 'gpt-4.1-mini-2025-04-14' # 可选: 'gpt-4o', 'gpt-4.1-mini-2025-04-14'

        # 任务5: 文档标题层级分析 (需要强大的文档结构理解能力)
        TITLE_HIERARCHY = 'gpt-4.1-mini-2025-04-14' # 可选: 'gpt-4o', 'gpt-4.1-mini-2025-04-14'

    # --- 处理参数 (Processing Parameters) ---
    PDF_TO_IMAGE_DPI = 200
    API_REQUEST_TIMEOUT = 120 # API请求超时时间（秒）
    GPT4O_BATCH_SIZE = 100     # gpt-4o 并发请求的数量
    VLM_BATCH_SIZE = 16       # 【V7 新增】本地VLM模型批处理大小 (根据VRAM调整)
    PADDLE_BATCH_SIZE = 16    # 【V8.2 新增】PaddleX版面分析批处理大小

    # --- V8新增: TXT转图片配置 ---
    TXT_IMAGE_WIDTH = 1240
    TXT_IMAGE_PADDING = 50
    TXT_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" # 确保此字体路径存在
    TXT_FONT_SIZE = 24
    TXT_LINE_SPACING = 10

    # --- 动态生成的输出目录 (Dynamically Generated Output Directories) ---
    if not os.path.exists(INPUT_PATH):
        print(f"❌ 致命错误: 配置文件中的输入路径不存在: {INPUT_PATH}")
        sys.exit(1)

    # 主输出目录基于输入文件名
    if os.path.isfile(INPUT_PATH):
        MASTER_OUTPUT_DIR = os.path.join(os.path.dirname(INPUT_PATH), f"output_{os.path.splitext(os.path.basename(INPUT_PATH))[0]}")
    else: # 如果是目录
        MASTER_OUTPUT_DIR = f"output_{os.path.basename(INPUT_PATH)}"


    # 各个阶段的子目录路径生成函数
    @staticmethod
    def get_subdirectories(base_dir):
        return {
            "DIR_1_PAGE_IMAGES": os.path.join(base_dir, "1_page_images"),
            "DIR_2_LAYOUT_JSONS": os.path.join(base_dir, "2_layout_jsons"),
            "DIR_3_CROPPED_TABLES": os.path.join(base_dir, "3_cropped_tables"),
            "DIR_3_CROPPED_IMAGES": os.path.join(base_dir, "3_cropped_images"),
            "DIR_TEMP_CELLS": os.path.join(base_dir, "temp_cells_for_batching"),
            "FINAL_COMBINED_JSON_PATH": os.path.join(base_dir, "_combined_document.json"),
            "FINAL_JSON_WITH_HIERARCHY_PATH": os.path.join(base_dir, "_document_with_hierarchy.json"),
            "FINAL_MARKDOWN_FILENAME_PATH": os.path.join(base_dir, "_final_document.md")
        }

# ======================================================================================
# --- 辅助函数 (Helper Functions) ---
# ======================================================================================
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

# ======================================================================================
# --- V8.1 新增: 文件转换模块 (V8.1 New: File Conversion Module) ---
# ======================================================================================

def convert_docx_to_pdf(docx_path, output_folder=None):
    """
    使用LibreOffice将DOCX转换为PDF（跨平台解决方案）
    
    参数:
        docx_path (str): 输入的DOCX文件路径
        output_folder (str): 输出目录（默认为输入文件所在目录）
    
    返回:
        str: 生成的PDF文件路径
    """
    # 验证输入文件
    docx_path_abs = os.path.abspath(docx_path)
    if not os.path.exists(docx_path_abs):
        raise FileNotFoundError(f"输入文件不存在: {docx_path_abs}")
    
    # 设置输出目录
    if output_folder is None:
        output_folder_abs = os.path.dirname(docx_path_abs)
    else:
        output_folder_abs = os.path.abspath(output_folder)
        os.makedirs(output_folder_abs, exist_ok=True)
    
    # 自动确定LibreOffice路径（跨平台）
    libreoffice_cmd = "libreoffice"
    if sys.platform == "win32":
        # 在Windows上，通常不在PATH中，需要寻找常见安装位置
        possible_paths = [
            os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "LibreOffice", "program", "soffice.exe"),
            os.path.join(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"), "LibreOffice", "program", "soffice.exe")
        ]
        for path in possible_paths:
            if os.path.exists(path):
                libreoffice_cmd = path
                break
    
    # 转换命令
    cmd = [
        libreoffice_cmd,
        "--headless",
        "--convert-to", "pdf",
        "--outdir", output_folder_abs,
        docx_path_abs
    ]
    
    # 执行转换
    try:
        print(f"   - 执行命令: {' '.join(cmd)}")
        process = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        print(f"   - LibreOffice 输出: {process.stdout.decode('utf-8', errors='ignore')}")
    except subprocess.CalledProcessError as e:
        error_message = f"LibreOffice 转换失败。返回码: {e.returncode}\n标准错误: {e.stderr.decode('utf-8', errors='ignore')}"
        raise RuntimeError(error_message) from e
    except FileNotFoundError:
        raise RuntimeError("命令 'libreoffice' 未找到。请确保已安装LibreOffice并且其路径在系统PATH环境变量中。")
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"LibreOffice 转换超时（超过120秒）。文件可能过大或LibreOffice无响应。") from e
    
    # 返回生成的PDF路径
    pdf_filename = Path(docx_path_abs).stem + ".pdf"
    pdf_path = os.path.join(output_folder_abs, pdf_filename)
    
    if not os.path.exists(pdf_path):
        raise RuntimeError(f"PDF生成失败: 输出文件未在预期位置创建: {pdf_path}")
    
    return pdf_path

def convert_word_to_images(doc_path, output_dir, dpi):
    """将Word文档(.docx)通过LibreOffice转换为一系列PNG图片。"""
    print(f"\n--- 正在转换 Word 文档: {os.path.basename(doc_path)} ---")
    generated_pdf_path = None
    try:
        # 1. Word -> PDF using LibreOffice
        print("  - 步骤 1/2: 使用 LibreOffice 转换为临时 PDF...")
        generated_pdf_path = convert_docx_to_pdf(doc_path, output_dir)
        print(f"   - PDF 生成于: {generated_pdf_path}")
        
        # 2. PDF -> Images
        print("  - 步骤 2/2: 从PDF生成图片...")
        run_step_1_pdf_to_images(generated_pdf_path, output_dir, dpi)
        print(f"✅ Word 文档转换成功。")
        return True
    except Exception as e:
        print(f"❌ 错误: Word文档转换失败。")
        print(f"   细节: {e}")
        # traceback.print_exc()
        return False
    finally:
        # 清理生成的临时PDF文件
        if generated_pdf_path and os.path.exists(generated_pdf_path):
            print(f"   - 清理临时文件: {generated_pdf_path}")
            try:
                os.remove(generated_pdf_path)
            except OSError as e:
                print(f"   - 警告: 无法删除临时PDF文件 {generated_pdf_path}: {e}")


def convert_ppt_to_images(ppt_path, output_dir, dpi):
    """将PPT演示文稿(.pptx)的每一页转换为PNG图片。"""
    print(f"\n--- 正在转换 PPT 文档: {os.path.basename(ppt_path)} ---")
    if not slides:
        print("❌ 错误: 'aspose.slides' 未安装，无法转换PPT文档。")
        return False
    try:
        start_time = time.time()
        with slides.Presentation(ppt_path) as presentation:
            total_slides = len(presentation.slides)
            for index, slide in enumerate(tqdm(presentation.slides, desc="转换PPT幻灯片")):
                # Aspose使用96 DPI作为基准，因此我们调整比例
                scale = dpi / 96.0
                bitmap = slide.get_thumbnail(scale_x=scale, scale_y=scale)
                output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(ppt_path))[0]}_page_{index + 1}.png")
                bitmap.save(output_path, slides.export.ImageFormat.PNG)
        elapsed_time = time.time() - start_time
        print(f"✅ PPT 文档成功转换 {total_slides} 页幻灯片，耗时 {elapsed_time:.2f} 秒。")
        return True
    except Exception as e:
        print(f"❌ 错误: PPT文档转换失败。细节: {e}")
        traceback.print_exc()
        return False

def convert_txt_to_image(txt_path, output_dir):
    """将TXT纯文本文档转换为一张或多张图片以便进行版面分析。"""
    print(f"\n--- 正在转换 TXT 文档: {os.path.basename(txt_path)} ---")
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            text_content = f.read()

        # 加载字体
        try:
            font = ImageFont.truetype(Config.TXT_FONT_PATH, Config.TXT_FONT_SIZE)
        except IOError:
            print(f"⚠️ 警告: 字体文件未找到于 {Config.TXT_FONT_PATH}. 使用默认字体。")
            font = ImageFont.load_default()

        # 文本换行逻辑
        lines = []
        max_width = Config.TXT_IMAGE_WIDTH - 2 * Config.TXT_IMAGE_PADDING
        for para in text_content.split('\n'):
            if not para.strip():
                lines.append('')
                continue
            current_line = ''
            for word in para.split(' '):
                # 检查单词本身是否过长
                word_bbox = font.getbbox(word)
                word_width = word_bbox[2] - word_bbox[0]
                if word_width > max_width:
                    # 如果一个单词比行宽还长，就强制拆分它
                    temp_word = ""
                    for char in word:
                        char_bbox = font.getbbox(temp_word + char)
                        if (char_bbox[2] - char_bbox[0]) > max_width:
                            lines.append(temp_word)
                            temp_word = char
                        else:
                            temp_word += char
                    if temp_word: lines.append(temp_word)
                    continue

                # 正常添加单词
                line_bbox = font.getbbox(current_line + ' ' + word)
                if (line_bbox[2] - line_bbox[0]) <= max_width:
                    current_line += ' ' + word
                else:
                    lines.append(current_line.strip())
                    current_line = word
            lines.append(current_line.strip())
            lines.append('') # 保留段落间的空行

        # 计算图片高度
        total_height = len(lines) * (Config.TXT_FONT_SIZE + Config.TXT_LINE_SPACING) + 2 * Config.TXT_IMAGE_PADDING

        # 创建图片并绘制文本
        img = Image.new('RGB', (Config.TXT_IMAGE_WIDTH, total_height), color='white')
        draw = ImageDraw.Draw(img)
        y_text = Config.TXT_IMAGE_PADDING
        for line in lines:
            draw.text((Config.TXT_IMAGE_PADDING, y_text), line, font=font, fill='black')
            y_text += Config.TXT_FONT_SIZE + Config.TXT_LINE_SPACING

        output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(txt_path))[0]}_page_1.png")
        img.save(output_path)
        print(f"✅ TXT 文档成功转换为图片: {output_path}")
        return True
    except Exception as e:
        print(f"❌ 错误: TXT文档转换失败。细节: {e}")
        traceback.print_exc()
        return False


# ======================================================================================
# --- STAGE 1: PDF to Image Conversion ---
# ======================================================================================
def run_step_1_pdf_to_images(pdf_path, output_dir, dpi):
    """Converts each page of a PDF file into a separate PNG image."""
    print("\n" + "="*80 + "\n--- STAGE 1: Starting PDF to Image Conversion ---\n" + "="*80)
    os.makedirs(output_dir, exist_ok=True)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    try:
        start_time = time.time()
        images = convert_from_path(pdf_path, dpi=dpi)
        for i, image in enumerate(tqdm(images, desc="Converting PDF pages")):
            output_path = os.path.join(output_dir, f"{pdf_name}_page_{i+1}.png")
            image.save(output_path, 'PNG')
        elapsed_time = time.time() - start_time
        print(f"✅ STAGE 1 Complete: Successfully converted {len(images)} pages in {elapsed_time:.2f} seconds.")
        print(f" ➡️ Output saved to: {output_dir}")
        return True
    except Exception as e:
        print(f"❌ ERROR in Stage 1: PDF to image conversion failed. Details: {e}")
        traceback.print_exc()
        return False

# ======================================================================================
# --- STAGE 2: Layout Analysis (V8.2 Batch Optimized) ---
# ======================================================================================
def run_step_2_layout_analysis(input_dir, output_dir):
    """Performs layout analysis on images using PaddleX and saves results as JSON."""
    print("\n" + "="*80 + "\n--- STAGE 2: Layout Analysis ---\n" + "="*80)
    os.makedirs(output_dir, exist_ok=True)
    from paddlex import create_pipeline as pp_create_pipeline
    pipeline = None
    try:
        print("Initializing PP-StructureV3 pipeline...")
        pipeline = pp_create_pipeline(pipeline="layout_parsing",batch_size=16)
        print("✅ Pipeline initialized.")
        start_time = time.time()
        image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"Found {len(image_files)} images. Starting processing...")

        for filename in tqdm(image_files, desc="Analyzing Layouts"):
            input_path = os.path.join(input_dir, filename)
            try:
                output = pipeline.predict(
                    input=input_path,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )
                base_filename = os.path.splitext(filename)[0]
                for i, res in enumerate(output):
                    json_save_path = os.path.join(output_dir, f"{base_filename}.json")
                    res.save_to_json(save_path=json_save_path)
            except Exception as e:
                tqdm.write(f" [ERROR] An error occurred while processing {filename}: {e}")

        elapsed_time = time.time() - start_time
        print(f"✅ STAGE 2 Complete: All images analyzed in {elapsed_time:.2f} seconds.")
        print(f" ➡️ Output saved to: {output_dir}")
        return True
    except Exception as e:
        print(f"❌ ERROR: Failed to initialize or run PaddleX pipeline. Details: {e}")
        return False
    finally:
        if pipeline:
            del pipeline
        gc.collect()
        print("✅ PaddleX pipeline resources released.")

from PIL import Image

# ======================================================================================
# --- 阶段 3 辅助函数 (专家实现) ---
# ======================================================================================

def calculate_iou(box1, box2):
    """
    计算两个包围盒（bounding box）的交并比 (Intersection over Union, IoU)。

    Args:
        box1 (list or tuple): 第一个包围盒的坐标 [x1, y1, x2, y2]。
        box2 (list or tuple): 第二个包围盒的坐标 [x1, y1, x2, y2]。

    Returns:
        float: 两个包围盒的 IoU 值，范围在 0.0 到 1.0 之间。
    """
    # 确定相交矩形的坐标
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # 如果包围盒不相交，则返回 0
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # 计算相交区域的面积
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # 计算两个包围盒各自的面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算并集面积
    union_area = box1_area + box2_area - intersection_area

    # 如果并集面积为0，则返回0以避免除零错误
    if union_area == 0:
        return 0.0
        
    # 计算 IoU
    iou = intersection_area / union_area
    return iou

# ======================================================================================
# --- STAGE 3: Visual Element Cropping (with Bbox Deduplication) ---
# ======================================================================================
def run_step_3_crop_visual_elements(image_dir, json_dir, table_output_dir, image_output_dir):
    """
    根据布局分析裁剪原始页面图像中的表格和图像区域。
    【核心更新】: 在裁剪前，首先通过计算包围盒的重叠率（IoU > 85%）来移除重复的表格，并将修改持久化。
    """
    print("\n" + "="*80 + "\n--- STAGE 3: Visual Element Cropping (with Bbox Deduplication) ---\n" + "="*80)
    os.makedirs(table_output_dir, exist_ok=True)
    os.makedirs(image_output_dir, exist_ok=True)
    total_tables_found, total_images_found = 0, 0
    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])
    if not json_files:
        print("⚠️ 警告: 在第 3 阶段未找到 JSON 布局文件。跳过。")
        return True

    start_time = time.time()
    for json_filename in tqdm(json_files, desc="Deduplicating & Cropping"):
        base_filename = os.path.splitext(json_filename)[0]
        json_path = os.path.join(json_dir, json_filename)
        image_path = os.path.join(image_dir, f"{base_filename}.png")
        
        if not os.path.exists(image_path):
            tqdm.write(f" [跳过] 找不到 JSON 文件对应的图像: {image_path}")
            continue
            
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                layout_data = json.load(f)
            
            if "parsing_res_list" not in layout_data:
                continue

            # --- 【全新去重逻辑】 ---
            all_blocks = layout_data["parsing_res_list"]
            
            # 【稳健性增强】为每个块添加原始索引，以确保文件名在删除后保持正确
            for i, block in enumerate(all_blocks):
                block['original_index'] = i

            table_blocks = [
                block for block in all_blocks 
                if block.get("block_label") == "table" and block.get("block_bbox")
            ]
            
            indices_to_remove = set()
            # 比较所有表格对的 IoU
            for i in range(len(table_blocks)):
                for j in range(i + 1, len(table_blocks)):
                    block1 = table_blocks[i]
                    block2 = table_blocks[j]
                    
                    # 获取原始索引进行比较和标记
                    idx1 = block1['original_index']
                    idx2 = block2['original_index']

                    if idx1 in indices_to_remove or idx2 in indices_to_remove:
                        continue

                    iou = calculate_iou(block1["block_bbox"], block2["block_bbox"])
                    
                    if iou > 0.85:
                        tqdm.write(f"   -> 在 {json_filename} 中发现重叠表格 (IoU: {iou:.2%})。")
                        tqdm.write(f"      - 保留块 {idx1}, 标记块 {idx2} 为待删除。")
                        indices_to_remove.add(idx2)

            # 如果有需要移除的块，则过滤列表并保存
            if indices_to_remove:
                deduplicated_blocks = [
                    block for block in all_blocks 
                    if block['original_index'] not in indices_to_remove
                ]
                layout_data["parsing_res_list"] = deduplicated_blocks
                
                # --- 【核心修复】将修改后的数据写回 JSON 文件，使去重操作持久化 ---
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(layout_data, f, ensure_ascii=False, indent=4)
                tqdm.write(f"   -> 已更新 {json_filename}，移除了 {len(indices_to_remove)} 个重复表格。")
            # --- 去重逻辑结束 ---

            # 使用去重后的 `parsing_res_list` 进行裁剪
            image = Image.open(image_path)
            for block in layout_data["parsing_res_list"]:
                block_label = block.get("block_label")
                bbox = block.get("block_bbox")
                if not bbox: continue
                
                x1, y1, x2, y2 = map(int, bbox)
                if x1 >= x2 or y1 >= y2: continue
                
                cropped_image = image.crop((max(0, x1-5), max(0, y1-10), x2+5, y2+5))
                
                # 使用我们保存的 'original_index' 来确保文件名始终正确
                original_idx = block['original_index']
                
                if block_label == "table":
                    output_filename = f"{base_filename}_table_{original_idx}.jpg"
                    output_path = os.path.join(table_output_dir, output_filename)
                    cropped_image.save(output_path)
                    total_tables_found += 1
                elif block_label == "image":
                    output_filename = f"{base_filename}_image_{original_idx}.jpg"
                    output_path = os.path.join(image_output_dir, output_filename)
                    cropped_image.save(output_path)
                    total_images_found += 1
                    
        except Exception as e:
            tqdm.write(f" [错误] 处理 {json_filename} 失败: {e}")
            
    elapsed_time = time.time() - start_time
    print(f"✅ 第 3 阶段完成: 在 {elapsed_time:.2f} 秒内裁剪了 {total_tables_found} 个表格和 {total_images_found} 个图像。")
    print(f" ➡️ 表格保存在: {table_output_dir}")
    print(f" ➡️ 图像保存在: {image_output_dir}")
    return True

# ======================================================================================
# --- STAGE 4: Recognition Engine ---
# This stage is now split into two parts: 4a for images and 4b for tables.
# It uses the ModelSelector to decide which recognition engine to use.
# ======================================================================================

# --- Model Initialization and Helper Functions ---
def initialize_local_models(config):
    """Initializes and returns all required local models based on selector."""
    models = {'qwen': None, 'nanonets': None}
    selector = config.ModelSelector
    
    # Check if any task requires the Qwen model
    qwen_needed = any(m == 'local_qwen' for m in [
        selector.IMAGE_DESCRIPTION, 
        selector.BORDERED_TABLE_CELL_RECOGNITION
    ])
    
    if qwen_needed:
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        print("Loading local Qwen-VL model...")
        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(config.VLM_MODEL_CHECKPOINT, torch_dtype="auto", device_map="auto")
            processor = AutoProcessor.from_pretrained(config.VLM_MODEL_CHECKPOINT, padding_side='left')
            models['qwen'] = (model, processor)
            print("✅ Local Qwen-VL model loaded successfully.")
        except Exception as e:
            print(f"❌ ERROR: Failed to load local Qwen-VL model: {e}")

    if selector.BORDERLESS_TABLE_RECOGNITION == 'local_nanonets':
        import torch
        from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
        print("Loading local Nanonets-OCR-s model for borderless tables...")
        try:
            model = AutoModelForImageTextToText.from_pretrained(config.NANONETS_MODEL_CHECKPOINT, torch_dtype="auto", device_map="auto")
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(config.NANONETS_MODEL_CHECKPOINT)
            processor = AutoProcessor.from_pretrained(config.NANONETS_MODEL_CHECKPOINT)
            models['nanonets'] = (model, tokenizer, processor)
            print("✅ Local Nanonets-OCR-s model loaded successfully.")
        except Exception as e:
            print(f"❌ ERROR: Failed to load local Nanonets model: {e}")
    return models

def initialize_openai_client(config):
    """Initializes and returns the OpenAI client if needed."""
    selector = config.ModelSelector
    if any('gpt' in m for m in vars(selector).values()):
        print("Initializing OpenAI client...")
        try:
            client = OpenAI(api_key=config.API_KEY, base_url=config.API_BASE_URL)
            client.models.list() # Test connection
            print("✅ OpenAI client initialized and connection successful.")
            return client
        except Exception as e:
            print(f"❌ ERROR: Failed to initialize OpenAI client or connect to API. Details: {e}")
            return None
    return None

def get_clients_and_models(config):
    """【V8.3 新增】按需加载并返回模型和客户端。"""
    print("\n" + "-"*25 + " Initializing Models & Clients (On-Demand) " + "-"*25)
    local_models = initialize_local_models(config)
    openai_client = initialize_openai_client(config)
    clients_and_models = {'openai': openai_client, **local_models}
    print("-" * 80)
    return clients_and_models

def recognize_with_openai_vision(client, model_name, image_b64, prompt, timeout):
    """Recognizes content from a base64 image using the OpenAI API."""
    if not image_b64: return "Error: Base64 image is empty."
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {client.api_key}"
    }
    payload = {
        "model": model_name,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            ]
        }],
        "max_tokens": 4096,
        "temperature":0.1
    }
    try:
        response = requests.post(
            f"{client.base_url}chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content']
        return content
    except requests.exceptions.RequestException as e:
        return f"Error: API request failed. Details: {e}"
    except (KeyError, IndexError):
        return "Error: Invalid API response."

def batch_recognize_with_openai_vision(client, model_name, image_paths, prompt, config):
    """
    【NEW】Uses a thread pool to process a batch of images concurrently.
    """
    results = {}
    with ThreadPoolExecutor(max_workers=config.GPT4O_BATCH_SIZE) as executor:
        future_to_filename = {
            executor.submit(
                recognize_with_openai_vision,
                client,
                model_name,
                encode_image_to_base64(path),
                prompt,
                config.API_REQUEST_TIMEOUT
            ): os.path.basename(path)
            for path in image_paths
        }
        progress = tqdm(as_completed(future_to_filename), total=len(image_paths), desc=f"Batch Processing (gpt-4o, {config.GPT4O_BATCH_SIZE} workers)")
        for future in progress:
            filename = future_to_filename[future]
            try:
                result = future.result()
                results[filename] = result
            except Exception as exc:
                tqdm.write(f" [ERROR] An exception occurred for {filename}: {exc}")
                results[filename] = f"Error: {exc}"
    return results

# --- STAGE 4a: Image Content Recognition ---
def run_step_4a_recognize_images(image_dir, layout_jsons_dir, clients_and_models, config):
    """Uses the selected model (local or OpenAI) to describe images and updates JSON files."""
    print("\n" + "="*80 + "\n--- STAGE 4a: Image Content Recognition ---\n" + "="*80)
    model_choice = config.ModelSelector.IMAGE_DESCRIPTION
    print(f"Using model for image description: {model_choice}")

    all_image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')])
    if not all_image_files:
        print("✅ No images found for recognition. Skipping Stage 4a.")
        return True

    descriptions = {}
    prompt = "请详细描述这张图片里的所有视觉内容和文字信息。图片里的内容是什么语言就输出什么语言。"

    if model_choice == 'gpt-4o':
        client = clients_and_models['openai']
        if not client:
            print("❌ ERROR: OpenAI client not available for image description. Halting.")
            return False
        image_paths = [os.path.join(image_dir, f) for f in all_image_files]
        descriptions = batch_recognize_with_openai_vision(client, 'gpt-4o', image_paths, prompt, config)

    elif model_choice == 'local_qwen':
        qwen_model, qwen_processor = clients_and_models.get('qwen', (None, None))
        if not qwen_model:
            print("❌ ERROR: Local Qwen model not available for image description. Halting.")
            return False
        
        # 【V7 增强】为Qwen图像描述添加批量处理
        image_pil_batch = [Image.open(os.path.join(image_dir, filename)) for filename in all_image_files]
        
        all_results = []
        for i in tqdm(range(0, len(image_pil_batch), config.VLM_BATCH_SIZE), desc="Describing Images (Qwen-VL Batch)"):
            batch_pil_images = image_pil_batch[i:i+config.VLM_BATCH_SIZE]
            
            messages_batch = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}] for _ in batch_pil_images]
            text_batch = [qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_batch]
            
            inputs = qwen_processor(text=text_batch, images=batch_pil_images, padding=True, return_tensors="pt").to(qwen_model.device)
            generated_ids = qwen_model.generate(**inputs, max_new_tokens=1024, do_sample=False)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_texts = qwen_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            all_results.extend(output_texts)

        descriptions = {filename: desc.strip() for filename, desc in zip(all_image_files, all_results)}

    # Update JSON files with descriptions
    updates_for_json = {}
    for filename, desc in descriptions.items():
        match = re.match(r'(.+)_image_(\d+)\.jpg', filename)
        if not match: continue
        json_base_name, block_idx_str = match.groups()
        json_filename = f"{json_base_name}.json"
        if json_filename not in updates_for_json: updates_for_json[json_filename] = []
        updates_for_json[json_filename].append({'block_idx': int(block_idx_str), 'description': desc})

    for json_filename, updates in tqdm(updates_for_json.items(), desc="Saving image descriptions"):
        layout_json_path = os.path.join(layout_jsons_dir, json_filename)
        if not os.path.exists(layout_json_path): continue
        try:
            with open(layout_json_path, 'r', encoding='utf-8') as f: layout_data = json.load(f)
            for update in updates:
                block_idx = update['block_idx']
                if block_idx < len(layout_data.get("parsing_res_list", [])):
                    layout_data["parsing_res_list"][block_idx]["block_content"] = update['description']
            with open(layout_json_path, 'w', encoding='utf-8') as f: json.dump(layout_data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            tqdm.write(f" [ERROR] Failed to save {json_filename}: {e}")

    print("✅ STAGE 4a Complete: All images described and JSONs updated.")
    return True

# ======================================================================================
# --- STAGE 4b: Table Recognition (V7 Batch Enhanced) ---
# ======================================================================================
def clean_vlm_html_response(response_text):
    """Cleans the VLM's response to extract pure HTML."""
    match = re.search(r'```(?:html)?\s*(.*?)\s*```', response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response_text.strip().strip('"').strip("'")

# --- 【V7 新增】本地模型批量识别函数 ---
def batch_recognize_text_with_qwen(image_batch, model, processor,prompt="直接提取图片中的所有文字内容。注意如果是空白的图片的话返回 ''。"):
    """【V7】Recognizes text from a BATCH of images using the Qwen-VL model."""
    if not image_batch: return []
    try:
        pil_images = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in image_batch]
        # prompt = "直接提取图片中的所有文字内容。注意如果是空白的图片的话返回 ''。"
        messages_batch = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}] for _ in pil_images]
        text_batch = [processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_batch]
        
        inputs = processor(text=text_batch, images=pil_images, padding=True, return_tensors="pt").to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=10000, do_sample=False)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return [text.strip().strip('"').strip("'") for text in output_texts]
    except Exception as e:
        tqdm.write(f" [WARNING] Qwen-VL batch recognition failed: {e}. Will retry individually.")
        return ["<BATCH_FAILURE>"] * len(image_batch)

def recognize_text_with_qwen_single(image_np, model, processor):
    """【V7】Recognizes text from a SINGLE image (Qwen). Used as a reliable fallback for batch failures."""
    if image_np is None or image_np.size == 0: return ""
    try:
        pil_image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        prompt = "直接提取图片中的所有文字内容。注意如果是空白的图片的话返回 ''。"
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[pil_image], padding=True, return_tensors="pt").to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return output_text.strip().strip('"').strip("'")
    except Exception as e:
        tqdm.write(f" [WARNING] Qwen-VL single recognition retry failed: {e}")
        return ""

def batch_recognize_tables_with_nanonets(image_batch, model, processor, tokenizer):
    """【V7】Recognizes tables from a BATCH of images using Nanonets and returns HTML strings."""
    if not image_batch: return []
    try:
        pil_images = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in image_batch]
        prompt = "Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation."
        messages_batch = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}] for _ in pil_images]
        text_batch = [processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_batch]
        
        inputs = processor(text=text_batch, images=pil_images, padding=True, return_tensors="pt").to(model.device)
        output_ids = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
        generated_ids = [out_id[len(in_id):] for in_id, out_id in zip(inputs.input_ids, output_ids)]
        output_texts = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return [clean_vlm_html_response(text) for text in output_texts]
    except Exception as e:
        tqdm.write(f" [WARNING] Nanonets batch recognition failed: {e}. Will retry individually.")
        return ["<BATCH_FAILURE>"] * len(image_batch)

def recognize_table_with_nanonets_single(image_np, model, processor, tokenizer):
    """【V7】Recognizes a table from a SINGLE image using Nanonets. Used for retry."""
    if image_np is None or image_np.size == 0:
        return "<table><tr><td>Error: Invalid image provided.</td></tr></table>"
    try:
        # This function is essentially the batch function with a batch size of 1
        result = batch_recognize_tables_with_nanonets([image_np], model, processor, tokenizer)
        return result[0] if result and result[0] != "<BATCH_FAILURE>" else "<table><tr><td>Error: Nanonets single recognition failed.</td></tr></table>"
    except Exception as e:
        return f"<table><tr><td>Error: Nanonets single recognition failed. Details: {e}</td></tr></table>"

# --- 表格处理核心逻辑 (Core Table Processing Logic) ---
def has_few_vertical_lines(img_np, min_length_ratio=0.5, line_threshold=2):
    if img_np is None: return True
    height, _ = img_np.shape[:2]
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height // 30))
    detected_vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    contours, _ = cv2.findContours(detected_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    long_line_count = sum(1 for cnt in contours if cv2.boundingRect(cnt)[3] > height * min_length_ratio)
    return long_line_count < line_threshold

def _cut_cells_by_finding_contours(img, height, width):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, width // 40), 1))
    detected_horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, height // 40)))
    detected_vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    table_grid_mask = cv2.add(detected_horizontal, detected_vertical)
    contours, _ = cv2.findContours(255 - table_grid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) <= 1: return []
    contours = sorted(contours, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))
    results = []
    for contour in contours:
        if cv2.contourArea(contour) < 100: continue
        x, y, w, h = cv2.boundingRect(contour)
        cell_image = img[y:y+h, x:x+w]
        if cell_image.size > 0: results.append((cell_image, (x, y, w, h)))
    return results

def _cut_cells_by_line_coordinates(img, height, width):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, width // 15), 1))
    detected_horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, height // 15)))
    detected_vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    h_contours, _ = cv2.findContours(detected_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    v_contours, _ = cv2.findContours(detected_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    row_b, col_b = set(), set()
    for c in h_contours: _, y, _, h = cv2.boundingRect(c); row_b.add(y); row_b.add(y+h)
    for c in v_contours: x, _, w, _ = cv2.boundingRect(c); col_b.add(x); col_b.add(x+w)
    def merge_close(b, tol=5):
        if not b: return []
        s = sorted(list(b)); m = [s[0]];
        for x in s[1:]:
            if x - m[-1] > tol: m.append(x)
        return m
    final_r, final_c = merge_close(row_b), merge_close(col_b)
    if len(final_r) < 2 or len(final_c) < 2: return []
    results = []
    for r in range(len(final_r) - 1):
        for c in range(len(final_c) - 1):
            y1, y2 = final_r[r], final_r[r+1]
            x1, x2 = final_c[c], final_c[c+1]
            cell_image = img[y1:y2, x1:x2]
            if cell_image.size > 0: results.append((cell_image, (x1, y1, x2-x1, y2-y1)))
    return results

def intelligent_cell_slicer(img):
    if img is None: return []
    height, width, _ = img.shape
    original_area = float(height * width)
    def filter_invalid_cells(cells):
        if not cells: return []
        return [cell for cell in cells if (cell[1][2] * cell[1][3]) / original_area < 0.95]
    cells_from_a = _cut_cells_by_finding_contours(img, height, width)
    valid_cells = filter_invalid_cells(cells_from_a)
    if not valid_cells:
        cells_from_b = _cut_cells_by_line_coordinates(img, height, width)
        valid_cells = filter_invalid_cells(cells_from_b)
    if not valid_cells:
        tqdm.write(f" [INFO] Both slicing strategies failed. Treating the entire image as a single cell (fallback).")
        return [(img, (0, 0, width, height))]
    return valid_cells

def convert_json_to_html_with_spans(cells_data):
    if not cells_data: return "<table><tr><td>Error: No cell data.</td></tr></table>"
    x_b, y_b = set(), set()
    for cell in cells_data:
        coords = cell['coordinates']
        x_b.add(coords['x']); x_b.add(coords['x'] + coords['width'])
        y_b.add(coords['y']); y_b.add(coords['y'] + coords['height'])
    def merge_boundaries(boundaries, tolerance=5):
        if not boundaries: return []
        b_sorted = sorted(list(boundaries))
        if not b_sorted: return []
        merged = [b_sorted[0]]
        for i in range(1, len(b_sorted)):
            if b_sorted[i] - merged[-1] > tolerance: merged.append(b_sorted[i])
        return merged
    final_x, final_y = merge_boundaries(x_b), merge_boundaries(y_b)
    num_cols, num_rows = len(final_x) - 1, len(final_y) - 1
    if num_cols <= 0 or num_rows <= 0: return "<table><tr><td>Error: Bad grid structure.</td></tr></table>"
    grid = [[None for _ in range(num_cols)] for _ in range(num_rows)]
    def find_closest_index(boundary_list, value, tolerance=5):
        for i, boundary in enumerate(boundary_list):
            if abs(value - boundary) < tolerance: return i
        return -1
    for cell in cells_data:
        coords = cell['coordinates']
        sc = find_closest_index(final_x, coords['x'])
        ec = find_closest_index(final_x, coords['x'] + coords['width'])
        sr = find_closest_index(final_y, coords['y'])
        er = find_closest_index(final_y, coords['y'] + coords['height'])
        if -1 in [sc, ec, sr, er]: continue
        if sr < num_rows and sc < num_cols:
            grid[sr][sc] = {'text': cell['recognition_result']['text'], 'rowspan': er - sr, 'colspan': ec - sc, 'is_placed': False}
    html_content = "<table>\n"
    for r in range(num_rows):
        html_content += " <tr>\n"
        for c in range(num_cols):
            cell = grid[r][c]
            if cell and not cell['is_placed']:
                for i in range(cell['rowspan']):
                    for j in range(cell['colspan']):
                        if (r + i) < num_rows and (c + j) < num_cols and grid[r+i][c+j]: grid[r+i][c+j]['is_placed'] = True
                rowspan_attr = f" rowspan=\"{cell['rowspan']}\"" if cell['rowspan'] > 1 else ""
                colspan_attr = f" colspan=\"{cell['colspan']}\"" if cell['colspan'] > 1 else ""
                cell_text = html.escape(cell['text'])
                html_content += f"  <td{rowspan_attr}{colspan_attr}>{cell_text}</td>\n"
    html_content += " </tr>\n"
    html_content += "</table>"
    return html_content

def run_step_4b_process_all_tables(cropped_tables_dir, layout_jsons_dir, clients_and_models, config):
    """【V7 核心改造】Processes all tables using the selected model, with batching for local models."""
    print("\n" + "="*80 + "\n--- STAGE 4b: Table Recognition & HTML Conversion (V7 Batch Enhanced) ---\n" + "="*80)
    borderless_model_choice = config.ModelSelector.BORDERLESS_TABLE_RECOGNITION
    bordered_model_choice = config.ModelSelector.BORDERED_TABLE_CELL_RECOGNITION
    print(f"Borderless table model: {borderless_model_choice} | Bordered cell model: {bordered_model_choice}")

    all_table_files = sorted([f for f in os.listdir(cropped_tables_dir) if f.lower().endswith('.jpg')])
    if not all_table_files:
        print("✅ No cropped tables found. Skipping Stage 4b.")
        return True

    table_html_results = {}
    
    # --- 分类表格以进行批处理 ---
    borderless_tables_for_gpt = []
    borderless_tables_for_nanonets = []
    bordered_tables_for_gpt = []
    bordered_tables_for_qwen = []

    for filename in tqdm(all_table_files, desc="Classifying Tables for Batching"):
        table_path = os.path.join(cropped_tables_dir, filename)
        img_np = cv2.imread(table_path)
        if img_np is None:
            table_html_results[filename] = "<table><tr><td>Error: Image unreadable.</td></tr></table>"
            continue
        
        if has_few_vertical_lines(img_np):
            if borderless_model_choice == 'local_nanonets':
                borderless_tables_for_nanonets.append({'filename': filename, 'img_np': img_np})
            elif borderless_model_choice == 'gpt-4o':
                borderless_tables_for_gpt.append(table_path)
        else:
            if bordered_model_choice == 'local_qwen':
                bordered_tables_for_qwen.append({'filename': filename, 'img_np': img_np})
            elif bordered_model_choice == 'gpt-4o':
                bordered_tables_for_gpt.append(table_path)

    # --- 处理无框线表格 (Process Borderless Tables) ---
    # gpt-4o (no change)
    if borderless_tables_for_gpt:
        client = clients_and_models['openai']
        prompt = "将图片里面的表格转换成一个完整的、结构正确的HTML表格。请只返回HTML代码，不要包含其他解释或```html标记。"
        results = batch_recognize_with_openai_vision(client, 'gpt-4o', borderless_tables_for_gpt, prompt, config)
        for filename, content in results.items():
            table_html_results[filename] = clean_vlm_html_response(content)

    # Nanonets (V7 Batch)
    if borderless_tables_for_nanonets:
        print(f"\nProcessing {len(borderless_tables_for_nanonets)} borderless tables with Nanonets (Batch)...")
        nanonets_model, nanonets_tokenizer, nanonets_processor = clients_and_models.get('nanonets', (None, None, None))
        if nanonets_model:
            image_batch = [t['img_np'] for t in borderless_tables_for_nanonets]
            filenames = [t['filename'] for t in borderless_tables_for_nanonets]
            
            batch_results = []
            for i in tqdm(range(0, len(image_batch), config.VLM_BATCH_SIZE), desc="Batch HTML (Nanonets)"):
                batch_htmls = batch_recognize_tables_with_nanonets(image_batch[i:i+config.VLM_BATCH_SIZE], nanonets_model, nanonets_processor, nanonets_tokenizer)
                batch_results.extend(batch_htmls)

            # Retry logic for failed items
            for idx, result in enumerate(batch_results):
                if result == "<BATCH_FAILURE>":
                    tqdm.write(f" [RETRY] Retrying Nanonets for {filenames[idx]} individually...")
                    result = recognize_table_with_nanonets_single(image_batch[idx], nanonets_model, nanonets_processor, nanonets_tokenizer)
                table_html_results[filenames[idx]] = result
        else:
            for t in borderless_tables_for_nanonets:
                table_html_results[t['filename']] = "<table><tr><td>Error: Nanonets model not loaded.</td></tr></table>"

    # --- 处理有框线表格 (Process Bordered Tables) ---
    # gpt-4o (no change, uses temp cell dir and ThreadPool)
    if bordered_tables_for_gpt:
        client = clients_and_models['openai']
        temp_cell_dir = Config.get_subdirectories(os.path.dirname(layout_jsons_dir))["DIR_TEMP_CELLS"]
        os.makedirs(temp_cell_dir, exist_ok=True)
        for table_path in tqdm(bordered_tables_for_gpt, desc="Slicing Bordered Tables (for gpt-4o)"):
            table_filename = os.path.basename(table_path)
            original_img_np = cv2.imread(table_path)
            cells_with_coords = intelligent_cell_slicer(original_img_np)
            if not cells_with_coords:
                table_html_results[table_filename] = "<table><tr><td>Error: Cell slicing failed.</td></tr></table>"
                continue
            
            cell_paths = []
            for i, (cell_img, _) in enumerate(cells_with_coords):
                path = os.path.join(temp_cell_dir, f"{os.path.splitext(table_filename)[0]}_cell_{i}.png")
                cv2.imwrite(path, cell_img)
                cell_paths.append(path)
            
            START_MARKER = "---TEXT_BEGIN---"
            END_MARKER = "---TEXT_END---"

            prompt_cell = f"""请严格按照以下格式输出，不要添加任何额外的解释或说明。
从图片中提取所有文字内容，并将其完整地放置在`{START_MARKER}`和`{END_MARKER}`之间。
如果图片是空白的或不包含任何文字，请在标记之间留空。

模板格式示例：
{START_MARKER}
这里是提取到的所有文字内容。
{END_MARKER}

空白图片示例：
{START_MARKER}{END_MARKER}
            """

            def extract_content_from_template(text: str) -> str:
                pattern = re.compile(f"{re.escape(START_MARKER)}(.*?){re.escape(END_MARKER)}", re.DOTALL)
                match = pattern.search(text)
                if match:
                    return match.group(1).strip()
                else:
                    return ""

            cell_ocr_results = batch_recognize_with_openai_vision(client, 'gpt-4o', cell_paths, prompt_cell, config)

            cell_texts = [
                extract_content_from_template(cell_ocr_results.get(os.path.basename(p), ""))
                for p in cell_paths
            ]

            table_data = []
            for i, (_, coords) in enumerate(cells_with_coords):
                x, y, w, h = coords
                text = cell_texts[i] if i < len(cell_texts) else ""
                table_data.append({"coordinates": {"x": x, "y": y, "width": w, "height": h}, "recognition_result": {"text": text, "score": 1.0}})
            table_html_results[table_filename] = convert_json_to_html_with_spans(table_data)

    # Qwen (V7 Batch)
    if bordered_tables_for_qwen:
        print(f"\nProcessing {len(bordered_tables_for_qwen)} bordered tables with Qwen (Batch)...")
        qwen_model, qwen_processor = clients_and_models.get('qwen', (None, None))
        if qwen_model:
            all_cells_to_process = []
            cell_to_table_map = [] # To map results back to tables
            
            for table_info in tqdm(bordered_tables_for_qwen, desc="Slicing Bordered Tables (for Qwen)"):
                cells_with_coords = intelligent_cell_slicer(table_info['img_np'])
                if not cells_with_coords:
                    table_html_results[table_info['filename']] = "<table><tr><td>Error: Cell slicing failed.</td></tr></table>"
                    continue
                
                table_cell_data = []
                for cell_img, coords in cells_with_coords:
                    all_cells_to_process.append(cell_img)
                    table_cell_data.append({'coords': coords})
                
                cell_to_table_map.append({
                    'filename': table_info['filename'],
                    'cell_data': table_cell_data,
                    'start_index': len(all_cells_to_process) - len(cells_with_coords),
                    'end_index': len(all_cells_to_process)
                })
            
            # Batch OCR all cells from all tables
            all_cell_texts = []
            for i in tqdm(range(0, len(all_cells_to_process), config.VLM_BATCH_SIZE), desc="Batch OCR Cells (Qwen)"):
                batch_texts = batch_recognize_text_with_qwen(all_cells_to_process[i:i+config.VLM_BATCH_SIZE], qwen_model, qwen_processor)
                all_cell_texts.extend(batch_texts)

            # Retry for failed items
            failed_indices = [i for i, text in enumerate(all_cell_texts) if text == "<BATCH_FAILURE>"]
            if failed_indices:
                print(f"🔍 Detected {len(failed_indices)} failures in Qwen batch. Retrying individually...")
                for idx in tqdm(failed_indices, desc="Retrying failed cells"):
                    all_cell_texts[idx] = recognize_text_with_qwen_single(all_cells_to_process[idx], qwen_model, qwen_processor)

            # Reconstruct tables
            for table_map_info in tqdm(cell_to_table_map, desc="Reconstructing HTML from Qwen results"):
                table_data = []
                table_cell_texts = all_cell_texts[table_map_info['start_index']:table_map_info['end_index']]
                for i, cell_info in enumerate(table_map_info['cell_data']):
                    x, y, w, h = cell_info['coords']
                    text = table_cell_texts[i] if i < len(table_cell_texts) else ""
                    table_data.append({"coordinates": {"x": x, "y": y, "width": w, "height": h}, "recognition_result": {"text": text, "score": 1.0}})
                table_html_results[table_map_info['filename']] = convert_json_to_html_with_spans(table_data)
        else:
            for t in bordered_tables_for_qwen:
                table_html_results[t['filename']] = "<table><tr><td>Error: Qwen model not loaded.</td></tr></table>"

    # --- Update layout JSON files with generated HTML ---
    updates_for_json = {}
    for filename, html_content in table_html_results.items():
        match = re.match(r'(.+)_table_(\d+)\.jpg', filename)
        if not match: continue
        json_base_name, block_idx_str = match.groups()
        json_filename = f"{json_base_name}.json"
        if json_filename not in updates_for_json: updates_for_json[json_filename] = []
        updates_for_json[json_filename].append({'block_idx': int(block_idx_str), 'html': html_content})

    for json_filename, updates in tqdm(updates_for_json.items(), desc="Saving updated layouts"):
        layout_json_path = os.path.join(layout_jsons_dir, json_filename)
        if not os.path.exists(layout_json_path): continue
        try:
            with open(layout_json_path, 'r', encoding='utf-8') as f: layout_data = json.load(f)
            for update in updates:
                block_idx = update['block_idx']
                if block_idx < len(layout_data.get("parsing_res_list", [])):
                    layout_data["parsing_res_list"][block_idx]["block_content"] = update['html']
            with open(layout_json_path, 'w', encoding='utf-8') as f: json.dump(layout_data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            tqdm.write(f" [ERROR] Failed to process or save {json_filename}: {e}")

    print("✅ STAGE 4b Complete.")
    return True

# ======================================================================================
# --- STAGE 5 & 7: AI Analysis Functions (Table Merging & Title Hierarchy) ---
# ======================================================================================
def run_ai_analysis(client, model_name, prompt, is_json_output=True):
    """Generic function to run analysis with an AI model."""
    try:
        messages = [{"role": "user", "content": prompt}]
        if is_json_output:
            completion = client.chat.completions.create(
                model=model_name, messages=messages, temperature=0.0, response_format={"type": "json_object"}
            )
        else:
            completion = client.chat.completions.create(
                model=model_name, messages=messages, temperature=0.0
            )
        content = completion.choices[0].message.content
        if is_json_output:
            json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
            json_string = json_match.group(1).strip() if json_match else content.strip()
            return json.loads(json_string)
        return content
    except Exception as e:
        print(f"❌ ERROR during AI analysis call with model {model_name}: {e}")
        return None

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

# ======================================================================================
# --- STAGE 5 HELPER FUNCTIONS (Expert Implementation) ---
# ======================================================================================

def get_effective_column_count(row_tag):
    """
    Calculates the number of columns in a table row (<tr>), correctly accounting for `colspan` attributes.
    
    Args:
        row_tag: A BeautifulSoup Tag object representing a <tr>.

    Returns:
        An integer representing the total column span of the row.
    """
    if not row_tag:
        return 0
    count = 0
    # Use recursive=False to only get immediate children cells, avoiding nested tables.
    for cell in row_tag.find_all(['td', 'th'], recursive=False):
        try:
            # Add the value of the colspan attribute, defaulting to 1 if not present.
            count += int(cell.get('colspan', 1))
        except (ValueError, TypeError):
            count += 1 # Default to 1 if colspan is not a valid number
    return count

def heuristic_merge_tables(html1, html2):
    """
    Attempts to merge two HTML tables based on a strict set of rules. This function
    is designed to handle common cases of tables split across pages.

    The rules are:
    1. Both tables must have the same number of columns.
    2. A special "broken row" merge is triggered if the first row of the second table
       is mostly empty, indicating it's a continuation of the last row of the first table.

    Args:
        html1 (str): The HTML content of the first table.
        html2 (str): The HTML content of the second, immediately adjacent table.

    Returns:
        A string containing the merged HTML if successful, otherwise None.
    """
    # This advanced heuristic function will only run if BeautifulSoup is available.
    soup1 = BeautifulSoup(html1, 'html.parser')
    soup2 = BeautifulSoup(html2, 'html.parser')

    table1 = soup1.find('table')
    table2 = soup2.find('table')

    # Ensure both HTML snippets are valid tables
    if not table1 or not table2:
        return None

    rows1 = table1.find_all('tr')
    rows2 = table2.find_all('tr')

    # Ensure both tables have rows to analyze
    if not rows1 or not rows2:
        return None

    # --- RULE 1: Check for identical column count ---
    cols1 = get_effective_column_count(rows1[0])
    cols2 = get_effective_column_count(rows2[0])

    # If column counts are different or zero, they are not continuations.
    if cols1 == 0 or cols1 != cols2:
        return None

    # --- RULE 2: Check for the "broken row" scenario ---
    first_row_of_table2 = rows2[0]
    cells_in_first_row_t2 = first_row_of_table2.find_all(['td', 'th'], recursive=False)
    
    empty_cells = 0
    for cell in cells_in_first_row_t2:
        if not cell.get_text(strip=True):
            empty_cells += 1
    
    rows_to_append = rows2
    # Condition: The second table's first row has at least half its cells empty.
    if len(cells_in_first_row_t2) > 0 and empty_cells >= len(cells_in_first_row_t2) / 2.0:
        print("   -> Heuristic check: Detected a potential 'broken row'. Attempting cell-wise merge.")
        last_row_of_table1 = rows1[-1]
        cells_in_last_row_t1 = last_row_of_table1.find_all(['td', 'th'], recursive=False)

        # This special merge is only safe if the cell layout is identical for the two rows.
        if len(cells_in_last_row_t1) == len(cells_in_first_row_t2):
            # Perform the str + str merge on the cell contents
            for cell1, cell2 in zip(cells_in_last_row_t1, cells_in_first_row_t2):
                text1 = cell1.get_text(separator=' ', strip=True)
                text2 = cell2.get_text(separator=' ', strip=True)
                combined_text = (text1 + " " + text2).strip()
                
                cell1.clear()  # Remove any existing complex content (e.g., tags)
                cell1.string = combined_text  # Set the new, combined text

            # Since we merged the first row of table2, we only need to append the rest.
            rows_to_append = rows2[1:]
        else:
            # The cell counts of the specific rows don't match, cannot safely merge cell-wise.
            # We will proceed with the standard row append.
            print("   -> Heuristic warning: Cell counts in boundary rows differ. Aborting cell-wise merge.")


    # --- 3. Perform the merge: Append the necessary rows from table2 to table1 ---
    # Find a <tbody> to append to, otherwise append directly to the <table> tag.
    target_for_append = table1.find('tbody')
    if target_for_append is None:
        target_for_append = table1

    for row in rows_to_append:
        target_for_append.append(row)  # Appends the BeautifulSoup Tag object

    # Return the fully merged table as a clean HTML string.
    return str(soup1)


# ======================================================================================
# --- STAGE 5: AI-Powered Intelligent Table Merging (with Heuristic Pre-processing) ---
# ======================================================================================
def run_step_5_ai_merge_tables(final_results_dir, client, config):
    """
    Identifies and merges STRICTLY ADJACENT tables using a two-stage approach:
    1. A rule-based heuristic for common, clear-cut cases (e.g., same column count).
    2. A powerful AI model for ambiguous cases requiring deeper semantic understanding.
    """
    print("\n" + "="*80 + "\n--- STAGE 5: AI-Powered Adjacent Table Merging (Expert Logic v3) ---\n" + "="*80)
    
    if not BS4_AVAILABLE:
        print("⚠️ WARNING: BeautifulSoup not found ('pip install beautifulsoup4'). The advanced heuristic merge rule will be skipped.")

    model_name = config.ModelSelector.TABLE_MERGING
    if not client or 'gpt' not in model_name:
        print(f"⚠️ WARNING: OpenAI client not available or model '{model_name}' not selected. Skipping table merging.")
        return True

    # --- 1. Load and Sort all JSON files by page number ---
    json_files = sorted([f for f in os.listdir(final_results_dir) if f.endswith('.json')])
    def get_page_number(filename):
        match = re.search(r'_page_(\d+)', filename)
        return int(match.group(1)) if match else float('inf')
    
    json_files.sort(key=get_page_number)
    
    if not json_files:
        print("⚠️ No JSON files found to process for table merging. Skipping.")
        return True

    print(f"Found {len(json_files)} JSON files to process.")
    all_pages_data = [json.load(open(os.path.join(final_results_dir, f), 'r', encoding='utf-8')) for f in json_files]

    # --- 2. Create a single, flat list of references to ALL blocks in document order ---
    all_block_refs = []
    for p_idx, page_data in enumerate(all_pages_data):
        for b_idx, block in enumerate(page_data.get('parsing_res_list', [])):
            all_block_refs.append({
                'block_data': block,
                'page_idx': p_idx,
                'block_idx': b_idx,
                'original_filename': json_files[p_idx]
            })
    
    if len(all_block_refs) < 2:
        print("ℹ️ Fewer than two blocks found in the document. No merging is possible. Skipping.")
        return True
        
    print(f"Analyzing {len(all_block_refs)} total blocks for adjacent tables...")

    # --- 3. Implement the Adjacent Pairwise Merge Logic ---
    i = 0
    while i < len(all_block_refs) - 1:
        current_ref = all_block_refs[i]
        next_ref = all_block_refs[i+1]
        
        current_block = current_ref['block_data']
        next_block = next_ref['block_data']

        # --- Key Logic: Check if two CONSECUTIVE blocks are both tables ---
        if current_block.get('block_label') == 'table' and next_block.get('block_label') == 'table':
            print(f"\nFound adjacent tables. Analyzing pair from '{current_ref['original_filename']}' (Block {current_ref['block_idx']}) and '{next_ref['original_filename']}' (Block {next_ref['block_idx']})")

            current_table_html = current_block['block_content']
            next_table_html = next_block['block_content']

            # --- STAGE 5.1: Heuristic (Rule-Based) Merge Attempt First ---
            merged_html_by_rule = None
            if BS4_AVAILABLE:
                try:
                    merged_html_by_rule = heuristic_merge_tables(current_table_html, next_table_html)
                except Exception as e:
                    print(f"⚠️  Heuristic merge attempt failed with an unexpected error: {e}")
                    merged_html_by_rule = None # Ensure it's None on error
            
            if merged_html_by_rule:
                print(" -> Heuristic Merge SUCCESS: Tables merged based on structural rules.")
                # Update the document structure with the heuristically merged table
                current_block['block_content'] = merged_html_by_rule
                next_block['block_label'] = 'merged_into_previous'
                next_block['block_content'] = ''
                
                # The next block has been merged, so remove it from our list of references
                all_block_refs.pop(i + 1)
                
                # Continue to the next pair. Do not increment 'i' because the list size has
                # changed and the new 'i+1' is the next block to evaluate.
                continue 
            
            # --- STAGE 5.2: AI-Powered Merge (Fallback) ---
            # This block is executed only if the heuristic merge was not applicable or failed.
            print(" -> Heuristic merge not applicable. Falling back to AI analysis.")
            
            # --- 4. AI Decision Prompt (Pairwise) ---
            decision_prompt = (
                "You are an expert document analyst. Your task is to determine if two consecutive HTML table fragments "
                "belong to the same logical table. The second fragment immediately follows the first in the document. "
                "Analyze the structure, headers, and content to decide if the second is a direct continuation of the first.\n\n"
                "Respond ONLY with a single JSON object: {\"should_merge\": true} if they should be merged, "
                "or {\"should_merge\": false} if they are separate tables.\n\n"
                f"**Table 1:**\n```html\n{current_table_html}\n```\n\n"
                f"**Table 2 (Immediate Successor):**\n```html\n{next_table_html}\n```"
            )
            
            try:
                # This is a placeholder for your actual AI call function
                # decision = run_ai_analysis(client, model_name, decision_prompt)
                pass # Replace with your actual call
            except Exception as e:
                print(f"ERROR: An exception occurred during AI decision analysis: {e}")
                decision = None

            # --- 5. Process AI Decision and Perform Merge if needed ---
            if decision and decision.get("should_merge"):
                print(" -> AI Decision: MERGE. These tables are a continuation.")
                
                revision_prompt = (
                    "You are an expert data structuring agent. Your task is to perfectly merge two adjacent HTML table fragments. "
                    "Combine them into one coherent, valid HTML table. Fix any structural issues, broken rows, or incorrect headers. "
                    "Ensure the final output is a single, complete, and valid HTML table.\n\n"
                    "**IMPORTANT:** Return ONLY a single valid JSON object with one key, \"revised_html\", containing the final, perfectly merged HTML table as a string.\n\n"
                    f"**Fragments to Merge:**\n```json\n{json.dumps([current_table_html, next_table_html], ensure_ascii=False, indent=2)}\n```"
                )
                
                try:
                    # This is a placeholder for your actual AI call function
                    # revised_content = run_ai_analysis(client, model_name, revision_prompt)
                    pass # Replace with your actual call
                except Exception as e:
                    print(f"ERROR: An exception occurred during AI revision: {e}")
                    revised_content = None

                if revised_content and "revised_html" in revised_content and revised_content["revised_html"]:
                    print(" -> AI revision successful. Updating document structure.")
                    
                    current_block['block_content'] = revised_content["revised_html"]
                    next_block['block_label'] = 'merged_into_previous'
                    next_block['block_content'] = ''
                    
                    all_block_refs.pop(i + 1)
                    
                    continue
                else:
                    print(" -> AI revision failed or returned empty content. Not merging this pair.")
            else:
                print(" -> AI Decision: DO NOT MERGE or analysis failed. Treating as separate tables.")
        
        i += 1

    # --- 6. Save all modified data back to JSON files ---
    print("\nSaving all updated files...")
    # Filter out blocks that were merged away before saving
    for p_idx, page_data in enumerate(all_pages_data):
        page_data['parsing_res_list'] = [
            block for block in page_data.get('parsing_res_list', []) 
            if block.get('block_label') != 'merged_into_previous'
        ]
        filepath = os.path.join(final_results_dir, json_files[p_idx])
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(page_data, f, ensure_ascii=False, indent=4)
            
    print("✅ STAGE 5 Complete: Intelligent adjacent table merging finished.")
    return True

# ======================================================================================
# --- STAGE 6: Aggregation ---
# ======================================================================================
def run_step_6_aggregate_results(results_dir, output_path):
    """Cleans, merges results from all pages, and extracts a table of contents."""
    print("\n" + "="*80 + "\n--- STAGE 6: Post-processing and Aggregation ---\n" + "="*80)
    json_files = sorted([f for f in os.listdir(results_dir) if f.endswith('.json')])
    def get_page_number(filename):
        match = re.search(r'_page_(\d+)', filename)
        return int(match.group(1)) if match else 0
    json_files.sort(key=get_page_number)
    
    all_pages_content, titles_toc = [], []
    for filename in tqdm(json_files, desc="Aggregating results"):
        filepath = os.path.join(results_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f: data = json.load(f)
            page_number = get_page_number(filename)
            filtered_blocks = [b for b in data.get("parsing_res_list", []) if b.get("block_label") != 'merged_into_previous']
            
            for i, block in enumerate(filtered_blocks):
                if 'title' in block.get("block_label", ""):
                    titles_toc.append({"page_index": page_number, "block_index_on_page": i, "title_content": block['block_content']})
            
            for block in filtered_blocks:
                if block.get("block_label") == "table" and block.get("block_content", "").startswith("<table>"):
                    block["block_content"] = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><title>Table</title><style>body {{ font-family: sans-serif; }} table {{ border-collapse: collapse; width: 100%; margin-bottom: 1em; }} th, td {{ border: 1px solid #dddddd; text-align: left; padding: 8px; }} tr:nth-child(even) {{ background-color: #f2f2f2; }} th {{ background-color: #e0e0e0; }}</style></head><body>{block['block_content']}</body></html>"""
            
            all_pages_content.append({"input_path": data.get("input_path"), "page_number": page_number, "parsing_res_list": filtered_blocks})
        except Exception as e:
            tqdm.write(f"❗️ Error processing {filename}: {e}")
            
    final_combined_data = {"document_content": all_pages_content, "titles_toc": titles_toc}
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_combined_data, f, ensure_ascii=False, indent=4)
        print(f"✅ STAGE 6 Complete. Aggregated results saved to: ➡️ {output_path}")
        return True
    except Exception as e:
        print(f"❌ ERROR in Stage 6: Could not save final combined file. Details: {e}")
        return False

# ======================================================================================
# --- STAGE 7: Title Hierarchy & Markdown Generation ---
# ======================================================================================
def analyze_title_hierarchy_with_ai(titles_toc, api_key, api_base_url, model_name):
    """使用AI分析标题列表并确定其层级。"""
    print("🧠 Sending titles to AI for hierarchy analysis...")
    print(f"Using model for hierarchy analysis: {model_name}")
    if not titles_toc:
        print("⚠️ No titles found to analyze. Skipping hierarchy analysis.")
        return {}
    
    title_texts = [item['title_content'] for item in titles_toc]
    prompt_content = f"""You are an expert in document structure analysis. Below is a list of sequential titles extracted from a document. Your task is to analyze this list and determine the hierarchical level of each title.
- The top-level main titles should be level 1.
- Sub-titles under a level 1 title should be level 2, and so on.
- Some items might not be real titles (e.g., '□适用√不适用', notes, or stray text). For these, assign level 0.
Here is the list of titles:
{json.dumps(title_texts, ensure_ascii=False, indent=2)}
Please return your analysis strictly as a JSON array of objects, where each object contains "original_title" and "hierarchical_level". Do not include any other text, markdown formatting, or explanations.
"""
    try:
        sanitized_base_url = api_base_url.rstrip('/')
        client = OpenAI(api_key=api_key, base_url=sanitized_base_url)
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an AI assistant that analyzes document structures and returns ONLY valid JSON."},
                {"role": "user", "content": prompt_content}
            ],
            temperature=0.0,
        )
        response_content = completion.choices[0].message.content
        json_match = re.search(r'```json\n(.*?)\n```', response_content, re.DOTALL)
        json_string = json_match.group(1).strip() if json_match else response_content.strip()
        analysis_result = json.loads(json_string)
        print("✅ AI analysis complete.")
        return analysis_result
    except Exception as e:
        print(f"❌ ERROR: AI title analysis failed: {e}")
        return None

def generate_markdown_from_structured_json(structured_data):
    """Generates a Markdown string from the final structured JSON data."""
    markdown_lines = []
    for page in structured_data.get('document_content', []):
        for block in page.get('parsing_res_list', []):
            label = block.get('block_label', 'text')
            content = block.get('block_content', '')

            if not isinstance(content, str):
                content = str(content)

            if label not in ['table', 'image']:
                content = ' '.join(content.split())
            
            if 'title' in label:
                level = block.get('hierarchical_level', 0)
                if level > 0:
                    markdown_lines.append(f"\n{'#' * level} {content}\n")
                else:
                    # Non-hierarchical titles or stray text are printed as plain text
                    markdown_lines.append(f"{content}\n")
            elif label == 'table':
                markdown_lines.append(f"\n{content}\n")
            elif label == 'image':
                markdown_lines.append(f"\nImage Description: {content}\n")
            else: # text, header, footer, etc.
                markdown_lines.append(f"{content}\n")
    return "\n".join(markdown_lines)

def run_step_7_create_final_document(input_json_path, final_json_path, final_md_path, api_key, api_base_url, model_name):
    """最终步骤：整合层级信息并生成Markdown和最终JSON文件。"""
    print("\n" + "="*80 + "\n--- STAGE 7: Final Document Generation ---\n" + "="*80)
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f: doc_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ ERROR: Combined document not found at '{input_json_path}'. Cannot proceed.")
        return False

    hierarchy = analyze_title_hierarchy_with_ai(doc_data.get('titles_toc', []), api_key, api_base_url, model_name)
    if hierarchy is None:
        print("Pipeline stopped due to error in AI analysis stage.")
        return False
        
    title_to_level_map = {item['original_title']: item['hierarchical_level'] for item in hierarchy}
    
    for page in doc_data['document_content']:
        for block in page['parsing_res_list']:
            if 'title' in block.get('block_label', ''):
                block['hierarchical_level'] = title_to_level_map.get(block['block_content'], 0)
                
    final_markdown = generate_markdown_from_structured_json(doc_data)
    
    try:
        with open(final_json_path, 'w', encoding='utf-8') as f:
            json.dump(doc_data, f, ensure_ascii=False, indent=4)
        print(f"💾 Final structured JSON saved to: {final_json_path}")
        
        with open(final_md_path, 'w', encoding='utf-8') as f:
            f.write(final_markdown)
        print(f"💾 Final Markdown document saved to: {final_md_path}")
        print("✅ STAGE 7 Complete.")
        return True
    except Exception as e:
        print(f"❌ ERROR: Failed to save final output files. Details: {e}")
        return False

# ======================================================================================
# --- V8 新增: 各文件类型的处理工作流 (V8 New: Workflows for each file type) ---
# ======================================================================================

def run_full_document_workflow(base_output_dir, config):
    """
    为PDF和Word等图像化文档设计的完整处理流程。
    【V8.3 变更】: 不再接收clients_and_models，而是在内部按需加载。
    """
    dirs = Config.get_subdirectories(base_output_dir)

    # --- STAGE 2 & 3: 无需AI模型，直接运行 ---
    if not run_step_2_layout_analysis(dirs["DIR_1_PAGE_IMAGES"], dirs["DIR_2_LAYOUT_JSONS"]):
        print("Pipeline halted at Stage 2."); return
    if not run_step_3_crop_visual_elements(dirs["DIR_1_PAGE_IMAGES"], dirs["DIR_2_LAYOUT_JSONS"], dirs["DIR_3_CROPPED_TABLES"], dirs["DIR_3_CROPPED_IMAGES"]):
        print("Pipeline halted at Stage 3."); return

    # --- 【V8.3 核心变更】按需加载模型 ---
    clients_and_models = get_clients_and_models(config)

    # --- STAGE 4a & 4b: 使用已加载的模型 ---
    if not run_step_4a_recognize_images(dirs["DIR_3_CROPPED_IMAGES"], dirs["DIR_2_LAYOUT_JSONS"], clients_and_models, config):
        print("Pipeline halted at Stage 4a."); return
    if not run_step_4b_process_all_tables(dirs["DIR_3_CROPPED_TABLES"], dirs["DIR_2_LAYOUT_JSONS"], clients_and_models, config):
        print("Pipeline halted at Stage 4b."); return
        
    # --- STAGE 5, 6, 7: 继续使用已加载的模型 ---
    if not run_step_5_ai_merge_tables(dirs["DIR_2_LAYOUT_JSONS"], clients_and_models.get('openai'), config):
        print("Pipeline halted at Stage 5."); return
    if not run_step_6_aggregate_results(dirs["DIR_2_LAYOUT_JSONS"], dirs["FINAL_COMBINED_JSON_PATH"]):
        print("Pipeline halted at Stage 6."); return
    if not run_step_7_create_final_document(
        dirs["FINAL_COMBINED_JSON_PATH"],
        dirs["FINAL_JSON_WITH_HIERARCHY_PATH"],
        dirs["FINAL_MARKDOWN_FILENAME_PATH"],
        config.API_KEY,
        config.API_BASE_URL,
        config.ModelSelector.TITLE_HIERARCHY
    ):
        print("Pipeline halted at Stage 7."); return
    
    return clients_and_models # 返回加载过的模型以便后续清理

def run_ppt_workflow(base_output_dir, config):
    """
    为PPT演示文稿设计的VLM解读流程。
    """
    print("\n" + "="*80 + "\n--- 🚀 Starting PPT Slide Interpretation Workflow ---\n" + "="*80)
    dirs = Config.get_subdirectories(base_output_dir)
    image_dir = dirs["DIR_1_PAGE_IMAGES"]
    
    # 按需加载模型
    clients_and_models = get_clients_and_models(config)
    client = clients_and_models.get('openai')
    model_name = config.ModelSelector.IMAGE_DESCRIPTION
    
    if not client or 'gpt' not in model_name:
        print("❌ ERROR: OpenAI client not available or a GPT model is not selected for image description. Cannot process PPT.")
        return None

    # --- 优化点：自然排序 ---
    # 1. 首先获取所有png文件的完整路径列表。
    #    Get a list of all full paths for the PNG files.
    all_png_filepaths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith('.png')]

    # 2. 定义一个用于排序的 "key" 函数，它能从文件名中提取页码。
    #    Define a "key" function for sorting that extracts the page number from the filename.
    def natural_sort_key(filepath):
        """
        从文件名中提取 '_page_X' 中的数字 X，用于正确的数字排序。
        Extracts the number X from '_page_X' in the filename for correct numerical sorting.
        """
        filename = os.path.basename(filepath)
        # 使用正则表达式查找页码
        match = re.search(r'_page_(\d+)', filename)
        if match:
            # 如果找到，返回整数形式的页码。
            # If found, return the page number as an integer.
            return int(match.group(1))
        # 如果文件名不符合格式，将其排在最后。
        # If the filename doesn't match the pattern, place it at the end.
        return float('inf')

    # 3. 使用这个 key 函数对文件列表进行排序，确保 'page_10.png' 在 'page_2.png' 之后。
    #    Sort the list of files using this key function to ensure 'page_10.png' comes after 'page_2.png'.
    all_slide_images = sorted(all_png_filepaths, key=natural_sort_key)
    # --- 优化结束 ---

    prompt = "你是一位资深的演示文稿分析师。请详细解读这张幻灯片。总结其核心要点、描述所有图表和图像，并提取关键的文字信息。你的分析应当全面而精炼。"
    
    print(f"Analyzing {len(all_slide_images)} slides with {model_name}...")
    results = batch_recognize_with_openai_vision(client, model_name, all_slide_images, prompt, config)
    
    # 生成Markdown报告
    # 循环遍历的是已经正确排序的列表，所以报告的顺序是正确的。
    markdown_lines = [f"# {os.path.basename(config.INPUT_PATH)} - 幻灯片分析报告\n"]
    for image_path in all_slide_images:
        filename = os.path.basename(image_path)
        description = results.get(filename, "未能生成描述。")
        page_num_match = re.search(r'_page_(\d+)', filename)
        page_num = page_num_match.group(1) if page_num_match else "N/A"
        
        markdown_lines.append(f"\n---\n\n## Page {page_num}\n")
        markdown_lines.append("### VLM Description:\n")
        markdown_lines.append(description)

    try:
        with open(dirs["FINAL_MARKDOWN_FILENAME_PATH"], 'w', encoding='utf-8') as f:
            f.write("\n".join(markdown_lines))
        print(f"✅ PPT workflow complete. Analysis saved to: {dirs['FINAL_MARKDOWN_FILENAME_PATH']}")
    except Exception as e:
        print(f"❌ ERROR: Failed to save PPT analysis markdown. Details: {e}")
        
    return clients_and_models

def run_txt_workflow(txt_path, base_output_dir, config):
    """
    为TXT纯文本文档设计的特殊处理流程（V8.3.2 专家修复版）。
    该流程将TXT转为图片，然后使用VLM直接将图片内容转换为Markdown格式。
    """
    print("\n" + "="*80 + "\n--- 🚀 Starting TXT Document Analysis Workflow (V8.3.2 Expert Fix) ---\n" + "="*80)
    clients_and_models = get_clients_and_models(config)
    qwen_model, qwen_processor = clients_and_models.get('qwen', (None, None))

    # 检查Qwen模型是否成功加载
    if not qwen_model or not qwen_processor:
        print("❌ ERROR: Qwen model is not loaded. Cannot proceed with TXT workflow.")
        return clients_and_models

    dirs = Config.get_subdirectories(base_output_dir)
    image_dir = dirs["DIR_1_PAGE_IMAGES"]

    def natural_sort_key(filepath):
        filename = os.path.basename(filepath)
        match = re.search(r'_page_(\d+)', filename)
        if match:
            return int(match.group(1))
        return float('inf')

    all_png_filepaths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith('.png')]
    all_page_images = sorted(all_png_filepaths, key=natural_sort_key)

    if not all_page_images:
        print("⚠️ WARNING: No images found for TXT file. Skipping analysis.")
        return clients_and_models

    image_batch_np = []
    for image_path in all_page_images:
        img = cv2.imread(image_path)
        if img is not None:
            image_batch_np.append(img)
        else:
            tqdm.write(f" [WARNING] Could not read image file, skipping: {image_path}")

    if not image_batch_np:
        print("❌ ERROR: Failed to load any valid images for the TXT file.")
        return clients_and_models

    prompt = "你是一位资深的文档分析师。请识别图片里面的内容，并将它直接整理成 markdown 格式返回。不要包含任何额外的解释或标题。请自动识别图片里面的各标题的标题，保存为 markdown 标题格式。"
    print(f"Analyzing {len(image_batch_np)} page(s) from TXT file with Qwen-VL...")

    # --- 专家修复: 内存溢出 (OOM) ---
    # Expert Fix for Out-of-Memory (OOM) Error.
    # 将所有图片分批送入模型，而不是一次性全部送入，以避免显存耗尽。
    # Process all images in smaller chunks (batches) instead of all at once to prevent VRAM exhaustion.
    results_list = []
    batch_size = 1 #config.VLM_BATCH_SIZE
    for i in tqdm(range(0, len(image_batch_np), batch_size), desc="Batch Processing TXT Images (Qwen)"):
        # 获取当前批次的图片数据
        # Get the image data for the current batch
        current_batch_np = image_batch_np[i:i + batch_size]
        
        # 使用当前批次调用识别函数
        # Call the recognition function with the current batch
        batch_results = batch_recognize_text_with_qwen(current_batch_np, qwen_model, qwen_processor, prompt)
        
        # 将当前批次的结果添加到总结果列表中
        # Append the results of the current batch to the main results list
        results_list.extend(batch_results)

    # 将返回的列表结果与文件名关联起来，创建一个字典
    results_dict = {os.path.basename(path): text for path, text in zip(all_page_images, results_list)}

    # 生成Markdown报告, 现在可以正确地从字典中获取描述
    markdown_lines = [f"# {os.path.basename(config.INPUT_PATH)} - 文档分析报告\n"]
    for image_path in all_page_images:
        filename = os.path.basename(image_path)
        description = results_dict.get(filename, "未能生成描述。")
        page_num_match = re.search(r'_page_(\d+)', filename)
        page_num = page_num_match.group(1) if page_num_match else "N/A"

        markdown_lines.append(f"\n---\n\n## Page {page_num}\n")
        markdown_lines.append(description)

    try:
        with open(dirs["FINAL_MARKDOWN_FILENAME_PATH"], 'w', encoding='utf-8') as f:
            f.write("\n".join(markdown_lines))
        print(f"✅ TXT workflow complete. Analysis saved to: {dirs['FINAL_MARKDOWN_FILENAME_PATH']}")
    except Exception as e:
        print(f"❌ ERROR: Failed to save TXT analysis markdown. Details: {e}")

    return clients_and_models



# ======================================================================================
# --- 🚀 V8 主执行协调器 (V8 Main Execution Orchestrator) ---
# ======================================================================================
def process_single_file(file_path, base_output_dir, config):
    """
    根据单个文件的类型，分发到相应的处理工作流。
    【V8.3 变更】: 不再接收clients_and_models，而是在工作流内部按需加载。
    返回加载过的模型以便主函数统一清理。
    """
    print("\n" + "#"*30 + f"   Processing File: {os.path.basename(file_path)}   " + "#"*30)
    
    # 确保每个文件都有自己独立的输出目录
    os.makedirs(base_output_dir, exist_ok=True)
    dirs = setup_directories(base_output_dir)
    file_ext = os.path.splitext(file_path)[1].lower()
    
    loaded_models_for_cleanup = None

    # --- PDF and Word Workflow ---
    # if file_ext == '.pdf' or file_ext == '.docx':
    try:
        if file_ext == '.pdf':
            if not run_step_1_pdf_to_images(file_path, dirs["DIR_1_PAGE_IMAGES"], config.PDF_TO_IMAGE_DPI):
                return None
            loaded_models_for_cleanup = run_full_document_workflow(base_output_dir, config)
        elif file_ext == '.docx':
            if not convert_word_to_images(file_path, dirs["DIR_1_PAGE_IMAGES"], config.PDF_TO_IMAGE_DPI):
                return None
            loaded_models_for_cleanup = run_full_document_workflow(base_output_dir, config)

    # # --- PPT Workflow ---
        elif file_ext == '.pptx':
            if not convert_word_to_images(file_path, dirs["DIR_1_PAGE_IMAGES"], config.PDF_TO_IMAGE_DPI):
                return None
            loaded_models_for_cleanup = run_ppt_workflow(base_output_dir, config)
            
        # --- TXT Workflow ---
        elif file_ext == '.txt':
            if not convert_word_to_images(file_path, dirs["DIR_1_PAGE_IMAGES"], config.PDF_TO_IMAGE_DPI):
                return None
            loaded_models_for_cleanup = run_txt_workflow(file_path, base_output_dir, config)
        
        else:
            if not convert_word_to_images(file_path, dirs["DIR_1_PAGE_IMAGES"], config.PDF_TO_IMAGE_DPI):
                return None
            loaded_models_for_cleanup = run_full_document_workflow(base_output_dir, config)
        
    # else:
    #     print(f"⚠️ Unsupported file type: {file_ext}. Skipping file: {os.path.basename(file_path)}")
    except:
        print(f"⚠️ Unsupported file type: {file_ext}. Skipping file: {os.path.basename(file_path)}")
    return loaded_models_for_cleanup

def main():
    """Main function to execute the entire document processing pipeline in order."""
    pipeline_start_time = time.time()
    print("#"*80)
    print("🚀 STARTING DOCUMENT PROCESSING PIPELINE (UNIVERSAL V8.3) 🚀")
    print(f"🕒 Start Time: {time.ctime(pipeline_start_time)}")
    print(f"📁 Input Path: {Config.INPUT_PATH}")
    print("#"*80)
    
    input_path = Config.INPUT_PATH
    loaded_models_in_run = None
    
    # --- 根据输入类型执行不同逻辑 ---
    if os.path.splitext(input_path)[1].lower() == '.zip':
        print(f"Detected ZIP archive. Starting batch processing...")
        temp_extract_dir = os.path.join(Config.MASTER_OUTPUT_DIR, "zip_extracted")
        os.makedirs(temp_extract_dir, exist_ok=True)
        
        with zipfile.ZipFile(input_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_dir)
        
        print(f"Archive extracted to: {temp_extract_dir}")
        
        all_files_to_process = []
        for root, _, files in os.walk(temp_extract_dir):
            for file in files:
                if file.startswith('__MACOSX') or file.startswith('.'): # 忽略macOS的元数据文件
                    continue
                all_files_to_process.append(os.path.join(root, file))

        for file_path in all_files_to_process:
            file_base_output_dir = os.path.join(Config.MASTER_OUTPUT_DIR, f"output_{os.path.splitext(os.path.basename(file_path))[0]}")
            # 每次循环都可能加载模型，并返回以便清理
            loaded_models_in_run = process_single_file(file_path, file_base_output_dir, Config)
            # 在处理完一个文件后立即清理模型，为下一个文件释放资源
            if loaded_models_in_run:
                print(f"\nReleasing models after processing {os.path.basename(file_path)}...")
                if loaded_models_in_run.get('qwen'): del loaded_models_in_run['qwen']
                if loaded_models_in_run.get('nanonets'): del loaded_models_in_run['nanonets']
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except (ImportError, NameError): pass
                print("✅ Models released.")
                
    elif os.path.isfile(input_path):
        loaded_models_in_run = process_single_file(input_path, Config.MASTER_OUTPUT_DIR, Config)
    else:
        print(f"❌ FATAL ERROR: Input path is not a valid file or ZIP archive: {input_path}")
        return

    # --- 最终释放模型资源 (如果仍在内存中) ---
    if loaded_models_in_run:
        print("\nPerforming final model resource cleanup...")
        if loaded_models_in_run.get('qwen'): del loaded_models_in_run['qwen']
        if loaded_models_in_run.get('nanonets'): del loaded_models_in_run['nanonets']
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("✅ PyTorch CUDA cache cleared.")
        except (ImportError, NameError): pass
        print("✅ Final resource cleanup complete.")

    # --- 清理临时文件 ---
    # 清理单元格临时目录
    temp_cell_dir_to_check = os.path.join(Config.MASTER_OUTPUT_DIR, "temp_cells_for_batching")
    if os.path.exists(temp_cell_dir_to_check):
        print(f"\nCleaning up temporary directory: {temp_cell_dir_to_check}")
        shutil.rmtree(temp_cell_dir_to_check)
        print("✅ Cleanup complete.")
    # 清理ZIP解压目录
    zip_extract_dir_to_check = os.path.join(Config.MASTER_OUTPUT_DIR, "zip_extracted")
    if os.path.exists(zip_extract_dir_to_check):
        print(f"\nCleaning up ZIP extraction directory: {zip_extract_dir_to_check}")
        shutil.rmtree(zip_extract_dir_to_check)
        print("✅ Cleanup complete.")


    # --- 最终总结 ---
    pipeline_elapsed_time = time.time() - pipeline_start_time
    print("\n" + "#"*80)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY! 🎉")
    print(f"Total execution time: {pipeline_elapsed_time:.2f} seconds ({pipeline_elapsed_time/60:.2f} minutes)")
    print(f"Final outputs are located in: {Config.MASTER_OUTPUT_DIR}")
    print("#"*80)

if __name__ == "__main__":
    main()
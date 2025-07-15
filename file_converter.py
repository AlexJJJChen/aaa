# -*- coding: utf-8 -*-
# ======================================================================================
# OMNI-PARSER: FILE CONVERTER MODULE
#
# 处理将各种文档格式（PDF, DOCX, PPTX, TXT）转换为图像序列。
# Handles converting various document formats (PDF, DOCX, PPTX, TXT) to image sequences.
# Logic is a 1:1 copy from the original omni_parser.py.
# ======================================================================================
import os
import sys
import time
import traceback
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from pdf2image import convert_from_path
from tqdm import tqdm

from config import Config

# 动态导入 aspose.slides
try:
    import aspose.slides as slides
except ImportError:
    print("❌ 警告: 'aspose-slides' 库未安装。处理PPT文档的功能将不可用。")
    print("   请运行: pip install aspose-slides")
    slides = None

def convert_to_images(file_path, output_dir, config):
    """
    将任何支持的文档类型转换为一系列PNG图片。
    这是一个分发函数，根据文件扩展名调用相应的转换器。
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    print(f"\n--- 正在转换 {file_ext.upper()} 文档: {os.path.basename(file_path)} ---")
    
    success = False
    if file_ext == '.pdf':
        success = _run_pdf_to_images(file_path, output_dir, config.PDF_TO_IMAGE_DPI)
    elif file_ext == '.docx':
        success = _convert_word_to_images(file_path, output_dir, config)
    elif file_ext == '.pptx':
        success = _convert_ppt_to_images(file_path, output_dir, config)
    elif file_ext == '.txt':
        success = _convert_txt_to_image(file_path, output_dir, config)
    else:
        print(f"⚠️ 不支持的文件类型: {file_ext}。跳过转换。")
        return False

    if success:
        print(f"✅ 文档转换成功。图片保存在: {output_dir}")
    else:
        print(f"❌ 文档转换失败。")
    return success

def _run_pdf_to_images(pdf_path, output_dir, dpi):
    """Converts each page of a PDF file into a separate PNG image."""
    print("\n" + "="*80 + "\n--- STAGE 1: Starting PDF to Image Conversion ---\n" + "="*80)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    try:
        start_time = time.time()
        images = convert_from_path(pdf_path, dpi=dpi)
        for i, image in enumerate(tqdm(images, desc="转换PDF页面")):
            output_path = os.path.join(output_dir, f"{pdf_name}_page_{i+1}.png")
            image.save(output_path, 'PNG')
        elapsed_time = time.time() - start_time
        print(f"✅ STAGE 1 完成: 在 {elapsed_time:.2f} 秒内成功转换 {len(images)} 页。")
        return True
    except Exception as e:
        print(f"❌ PDF到图片转换失败。细节: {e}")
        traceback.print_exc()
        return False

def _convert_word_to_images(doc_path, output_dir, config):
    """将Word文档通过LibreOffice转换为一系列PNG图片。"""
    generated_pdf_path = None
    try:
        print("  - 步骤 1/2: 使用 LibreOffice 转换为临时 PDF...")
        generated_pdf_path = _convert_docx_to_pdf(doc_path, output_dir)
        print(f"   - PDF 生成于: {generated_pdf_path}")
        
        print("  - 步骤 2/2: 从PDF生成图片...")
        return _run_pdf_to_images(generated_pdf_path, output_dir, config.PDF_TO_IMAGE_DPI)
    except Exception as e:
        print(f"   - 细节: {e}")
        return False
    finally:
        if generated_pdf_path and os.path.exists(generated_pdf_path):
            print(f"   - 清理临时文件: {generated_pdf_path}")
            try:
                os.remove(generated_pdf_path)
            except OSError as e:
                print(f"   - 警告: 无法删除临时PDF文件 {generated_pdf_path}: {e}")

def _convert_docx_to_pdf(docx_path, output_folder=None):
    """使用LibreOffice将DOCX转换为PDF（跨平台解决方案）。"""
    docx_path_abs = os.path.abspath(docx_path)
    if not os.path.exists(docx_path_abs):
        raise FileNotFoundError(f"输入文件不存在: {docx_path_abs}")
    
    output_folder_abs = os.path.abspath(output_folder or os.path.dirname(docx_path_abs))
    os.makedirs(output_folder_abs, exist_ok=True)
    
    libreoffice_cmd = "libreoffice"
    if sys.platform == "win32":
        possible_paths = [
            os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "LibreOffice", "program", "soffice.exe"),
            os.path.join(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"), "LibreOffice", "program", "soffice.exe")
        ]
        libreoffice_cmd = next((path for path in possible_paths if os.path.exists(path)), "libreoffice")

    cmd = [libreoffice_cmd, "--headless", "--convert-to", "pdf", "--outdir", output_folder_abs, docx_path_abs]
    
    try:
        print(f"   - 执行命令: {' '.join(cmd)}")
        process = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120, errors='ignore')
        print(f"   - LibreOffice 输出: {process.stdout}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"LibreOffice 转换失败。返回码: {e.returncode}\n标准错误: {e.stderr}") from e
    except FileNotFoundError:
        raise RuntimeError("命令 'libreoffice' 未找到。请确保已安装LibreOffice并且其路径在系统PATH环境变量中。")
    except subprocess.TimeoutExpired as e:
        raise RuntimeError("LibreOffice 转换超时。") from e
    
    pdf_path = os.path.join(output_folder_abs, Path(docx_path_abs).stem + ".pdf")
    if not os.path.exists(pdf_path):
        raise RuntimeError(f"PDF生成失败: {pdf_path}")
    
    return pdf_path

def _convert_ppt_to_images(ppt_path, output_dir, config):
    """将PPT演示文稿的每一页转换为PNG图片。"""
    if not slides:
        print("❌ 错误: 'aspose.slides' 未安装，无法转换PPT文档。")
        return False
    try:
        start_time = time.time()
        ppt_name = os.path.splitext(os.path.basename(ppt_path))[0]
        with slides.Presentation(ppt_path) as presentation:
            total_slides = len(presentation.slides)
            scale = config.PDF_TO_IMAGE_DPI / 96.0
            for index, slide in enumerate(tqdm(presentation.slides, desc="转换PPT幻灯片")):
                bitmap = slide.get_thumbnail(scale_x=scale, scale_y=scale)
                output_path = os.path.join(output_dir, f"{ppt_name}_page_{index + 1}.png")
                bitmap.save(output_path, slides.export.ImageFormat.PNG)
        elapsed_time = time.time() - start_time
        print(f"✅ 成功转换 {total_slides} 页幻灯片，耗时 {elapsed_time:.2f} 秒。")
        return True
    except Exception as e:
        print(f"❌ 错误: PPT文档转换失败。细节: {e}")
        traceback.print_exc()
        return False

def _convert_txt_to_image(txt_path, output_dir, config):
    """将TXT纯文本文档转换为一张或多张图片。"""
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            text_content = f.read()

        try:
            font = ImageFont.truetype(config.TXT_FONT_PATH, config.TXT_FONT_SIZE)
        except IOError:
            print(f"⚠️ 警告: 字体文件未找到。使用默认字体。")
            font = ImageFont.load_default()

        lines = []
        max_width = config.TXT_IMAGE_WIDTH - 2 * config.TXT_IMAGE_PADDING
        for para in text_content.split('\n'):
            if not para.strip():
                lines.append('')
                continue
            current_line = ''
            for word in para.split(' '):
                word_bbox = font.getbbox(word)
                word_width = word_bbox[2] - word_bbox[0]
                if word_width > max_width:
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
                line_bbox = font.getbbox(current_line + ' ' + word)
                if (line_bbox[2] - line_bbox[0]) <= max_width:
                    current_line += ' ' + word
                else:
                    lines.append(current_line.strip())
                    current_line = word
            lines.append(current_line.strip())
            lines.append('')

        total_height = len(lines) * (config.TXT_FONT_SIZE + config.TXT_LINE_SPACING) + 2 * config.TXT_IMAGE_PADDING
        img = Image.new('RGB', (config.TXT_IMAGE_WIDTH, total_height), color='white')
        draw = ImageDraw.Draw(img)
        y_text = config.TXT_IMAGE_PADDING
        for line in lines:
            draw.text((config.TXT_IMAGE_PADDING, y_text), line, font=font, fill='black')
            y_text += config.TXT_FONT_SIZE + config.TXT_LINE_SPACING

        output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(txt_path))[0]}_page_1.png")
        img.save(output_path)
        return True
    except Exception as e:
        print(f"❌ 错误: TXT文档转换失败。细节: {e}")
        traceback.print_exc()
        return False

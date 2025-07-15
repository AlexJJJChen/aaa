# -*- coding: utf-8 -*-
# ======================================================================================
# OMNI-PARSER: CONFIGURATION MODULE
#
# 集中管理所有路径、模型和API密钥的配置。
# Centrally manages all paths, models, and API keys.
# ======================================================================================
import os
import sys

class Config:
    """
    Configuration class to centrally manage all paths, models, and API keys.
    This structure is a 1:1 copy from the original omni_parser.py.
    """
    # --- V8核心变更: 输入文件路径 (V8 Core Change: Input File Path) ---
    # 支持 .pdf, .docx, .txt, .pptx, .zip
    INPUT_PATH = "/project/chenjian/bbb/[定期报告][2023-03-20][朗鸿科技]朗鸿科技2022年年度报告摘要.pdf" # 请修改为您的实际文件路径

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
        IMAGE_DESCRIPTION = 'gpt-4o'
        BORDERLESS_TABLE_RECOGNITION = 'local_nanonets'
        BORDERED_TABLE_CELL_RECOGNITION = 'local_qwen'
        TABLE_MERGING = 'gpt-4.1-mini-2025-04-14'
        TITLE_HIERARCHY = 'gpt-4.1-mini-2025-04-14'

    # --- 处理参数 (Processing Parameters) ---
    PDF_TO_IMAGE_DPI = 200
    API_REQUEST_TIMEOUT = 120
    GPT4O_BATCH_SIZE = 100
    VLM_BATCH_SIZE = 16
    PADDLE_BATCH_SIZE = 16

    # --- TXT转图片配置 ---
    TXT_IMAGE_WIDTH = 1240
    TXT_IMAGE_PADDING = 50
    # 确保此字体路径存在，或替换为您系统中的可用字体路径
    TXT_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" 
    TXT_FONT_SIZE = 24
    TXT_LINE_SPACING = 10

    # --- 动态生成的输出目录 (Dynamically Generated Output Directories) ---
    @staticmethod
    def get_master_output_dir(input_path):
        """
        Generates the main output directory based on the input file name.
        This logic is derived from the original omni_parser.py's main execution block.
        """
        if not os.path.exists(input_path):
            print(f"❌ 致命错误: 配置文件中的输入路径不存在: {input_path}")
            sys.exit(1)
            
        if os.path.isfile(input_path):
            return os.path.join(os.path.dirname(input_path), f"output_{os.path.splitext(os.path.basename(input_path))[0]}")
        else: # If it's a directory
            return os.path.join(os.path.dirname(input_path), f"output_{os.path.basename(input_path)}")

    @staticmethod
    def get_subdirectories(base_dir):
        """Generates a dictionary of all required subdirectories."""
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
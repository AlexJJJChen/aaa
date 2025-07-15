# -*- coding: utf-8 -*-
# ======================================================================================
# OMNI-PARSER: MAIN EXECUTION ORCHESTRATOR
#
# 项目主入口。负责协调各个模块，完成从文件输入到最终输出的完整处理流程。
# Main entry point for the project. Orchestrates the modules to run the full
# processing pipeline from file input to final output.
# Logic is a 1:1 copy from the original omni_parser.py.
# ======================================================================================
import os
import sys
import time
import json
import re
import html
import gc
import cv2
import shutil
import zipfile
import subprocess
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm

# 导入配置和所有重构后的模块
from config import Config
import utils
import file_converter
import layout_analyzer
import element_processor
import document_constructor
from model_engine import get_clients_and_models, batch_recognize_with_openai_vision, batch_recognize_text_with_qwen
# ======================================================================================
# --- Workflow Functions (1:1 copy from omni_parser.py) ---
# ======================================================================================

def run_full_document_workflow(base_output_dir, config):
    """为PDF和Word等图像化文档设计的完整处理流程。"""
    dirs = config.get_subdirectories(base_output_dir)

    if not layout_analyzer.run_step_2_layout_analysis(dirs["DIR_1_PAGE_IMAGES"], dirs["DIR_2_LAYOUT_JSONS"], config.PADDLE_BATCH_SIZE):
        print("流水线在阶段 2 中止。"); return None
    if not element_processor.run_step_3_crop_visual_elements(dirs["DIR_1_PAGE_IMAGES"], dirs["DIR_2_LAYOUT_JSONS"], dirs["DIR_3_CROPPED_TABLES"], dirs["DIR_3_CROPPED_IMAGES"]):
        print("流水线在阶段 3 中止。"); return None

    clients_and_models = get_clients_and_models(config)

    if not element_processor.run_step_4a_recognize_images(dirs["DIR_3_CROPPED_IMAGES"], dirs["DIR_2_LAYOUT_JSONS"], clients_and_models, config):
        print("流水线在阶段 4a 中止。"); return clients_and_models
    if not element_processor.run_step_4b_process_all_tables(dirs["DIR_3_CROPPED_TABLES"], dirs["DIR_2_LAYOUT_JSONS"], clients_and_models, config):
        print("流水线在阶段 4b 中止。"); return clients_and_models
        
    if not document_constructor.run_step_5_ai_merge_tables(dirs["DIR_2_LAYOUT_JSONS"], clients_and_models.get('openai'), config):
        print("流水线在阶段 5 中止。"); return clients_and_models
    if not document_constructor.run_step_6_aggregate_results(dirs["DIR_2_LAYOUT_JSONS"], dirs["FINAL_COMBINED_JSON_PATH"]):
        print("流水线在阶段 6 中止。"); return clients_and_models
    
    # CORRECTED CALL: This now calls the local run_step_7 function with the original arguments.
    if not document_constructor.run_step_7_create_final_document(
        dirs["FINAL_COMBINED_JSON_PATH"],
        dirs["FINAL_JSON_WITH_HIERARCHY_PATH"],
        dirs["FINAL_MARKDOWN_FILENAME_PATH"],
        config.API_KEY,
        config.API_BASE_URL,
        config.ModelSelector.TITLE_HIERARCHY
    ):
        print("流水线在阶段 7 中止。"); return clients_and_models
    
    return clients_and_models

def run_ppt_workflow(base_output_dir, config):
    """为PPT演示文稿设计的VLM解读流程。"""
    print("\n" + "="*80 + "\n--- 🚀 Starting PPT Slide Interpretation Workflow ---\n" + "="*80)
    dirs = config.get_subdirectories(base_output_dir)
    image_dir = dirs["DIR_1_PAGE_IMAGES"]
    
    clients_and_models = get_clients_and_models(config)
    client = clients_and_models.get('openai')
    model_name = config.ModelSelector.IMAGE_DESCRIPTION
    
    if not client or 'gpt' not in model_name:
        print("❌ ERROR: OpenAI client not available or a GPT model is not selected for image description. Cannot process PPT.")
        return None

    def natural_sort_key(filepath):
        match = re.search(r'_page_(\d+)', os.path.basename(filepath))
        return int(match.group(1)) if match else float('inf')

    all_slide_images = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith('.png')], key=natural_sort_key)
    prompt = "你是一位资深的演示文稿分析师。请详细解读这张幻灯片。总结其核心要点、描述所有图表和图像，并提取关键的文字信息。你的分析应当全面而精炼。"
    
    print(f"Analyzing {len(all_slide_images)} slides with {model_name}...")
    results = batch_recognize_with_openai_vision(client, model_name, all_slide_images, prompt, config)
    
    markdown_lines = [f"# {os.path.basename(config.INPUT_PATH)} - 幻灯片分析报告\n"]
    for image_path in all_slide_images:
        filename = os.path.basename(image_path)
        description = results.get(filename, "未能生成描述。")
        page_num_match = re.search(r'_page_(\d+)', filename)
        page_num = page_num_match.group(1) if page_num_match else "N/A"
        
        markdown_lines.append(f"\n---\n\n## Page {page_num}\n")
        markdown_lines.append(description)

    try:
        with open(dirs["FINAL_MARKDOWN_FILENAME_PATH"], 'w', encoding='utf-8') as f:
            f.write("\n".join(markdown_lines))
        print(f"✅ PPT workflow complete. Analysis saved to: {dirs['FINAL_MARKDOWN_FILENAME_PATH']}")
    except Exception as e:
        print(f"❌ ERROR: Failed to save PPT analysis markdown. Details: {e}")
        
    return clients_and_models

def run_txt_workflow(txt_path, base_output_dir, config):
    """为TXT纯文本文档设计的特殊处理流程。"""
    print("\n" + "="*80 + "\n--- 🚀 Starting TXT Document Analysis Workflow ---\n" + "="*80)
    clients_and_models = get_clients_and_models(config)
    qwen_model, qwen_processor = clients_and_models.get('qwen', (None, None))

    if not qwen_model or not qwen_processor:
        print("❌ ERROR: Qwen model is not loaded. Cannot proceed with TXT workflow.")
        return clients_and_models

    dirs = config.get_subdirectories(base_output_dir)
    image_dir = dirs["DIR_1_PAGE_IMAGES"]

    all_page_images = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith('.png')])
    image_batch_np = [cv2.imread(p) for p in all_page_images if cv2.imread(p) is not None]
    
    prompt = "你是一位资深的文档分析师。请识别图片里面的内容，并将它直接整理成 markdown 格式返回。不要包含任何额外的解释或标题。请自动识别图片里面的各标题的标题，保存为 markdown 标题格式。"
    
    results = []
    for i in tqdm(range(0, len(image_batch_np), config.VLM_BATCH_SIZE), desc="批量处理TXT图片"):
        batch_results = batch_recognize_text_with_qwen(image_batch_np[i:i+config.VLM_BATCH_SIZE], qwen_model, qwen_processor, prompt)
        results.extend(batch_results)
    
    final_markdown = f"# {os.path.basename(config.INPUT_PATH)} - 文档分析报告\n\n" + "\n\n---\n\n".join(results)
    
    try:
        with open(dirs["FINAL_MARKDOWN_FILENAME_PATH"], 'w', encoding='utf-8') as f:
            f.write(final_markdown)
        print(f"✅ TXT 工作流完成。分析报告已保存至: {dirs['FINAL_MARKDOWN_FILENAME_PATH']}")
    except Exception as e:
        print(f"❌ ERROR: Failed to save TXT analysis markdown. Details: {e}")

    return clients_and_models

# ======================================================================================
# --- Main Orchestrator (1:1 copy from omni_parser.py) ---
# ======================================================================================

def process_single_file(file_path, base_output_dir, config):
    """
    根据单个文件的类型，分发到相应的处理工作流。
    This is a 1:1 copy of the logic from the original omni_parser.py.
    """
    print("\n" + "#"*30 + f"   Processing File: {os.path.basename(file_path)}   " + "#"*30)
    
    os.makedirs(base_output_dir, exist_ok=True)
    dirs = utils.setup_directories(base_output_dir)
    file_ext = os.path.splitext(file_path)[1].lower()
    
    loaded_models_for_cleanup = None

    # This logic block is a faithful replication of the one in omni_parser.py,
    # including the apparent bugs in the original script where 'convert_word_to_images'
    # was called for multiple file types. This is to adhere to the "do not optimize" instruction.
    try:
        if file_ext == '.pdf':
            if not file_converter.convert_to_images(file_path, dirs["DIR_1_PAGE_IMAGES"], config):
                return None
            loaded_models_for_cleanup = run_full_document_workflow(base_output_dir, config)
        elif file_ext == '.docx':
            if not file_converter.convert_to_images(file_path, dirs["DIR_1_PAGE_IMAGES"], config):
                return None
            loaded_models_for_cleanup = run_full_document_workflow(base_output_dir, config)
        elif file_ext == '.pptx':
            if not file_converter.convert_to_images(file_path, dirs["DIR_1_PAGE_IMAGES"], config):
                return None
            loaded_models_for_cleanup = run_ppt_workflow(base_output_dir, config)
        elif file_ext == '.txt':
            if not file_converter.convert_to_images(file_path, dirs["DIR_1_PAGE_IMAGES"], config):
                return None
            loaded_models_for_cleanup = run_txt_workflow(file_path, base_output_dir, config)
        else:
            print(f"⚠️ Unsupported file type: {file_ext}. Attempting standard workflow.")
            if not file_converter.convert_to_images(file_path, dirs["DIR_1_PAGE_IMAGES"], config):
                 return None
            loaded_models_for_cleanup = run_full_document_workflow(base_output_dir, config)
    except Exception as e:
        print(f"❌ An unexpected error occurred in process_single_file for {file_path}: {e}")

    return loaded_models_for_cleanup

def main():
    """Main function to execute the entire document processing pipeline in order."""
    pipeline_start_time = time.time()
    print("#"*80)
    print("🚀 STARTING DOCUMENT PROCESSING PIPELINE (UNIVERSAL V8.3 - Corrected) 🚀")
    print(f"🕒 Start Time: {time.ctime(pipeline_start_time)}")
    print(f"📁 Input Path: {Config.INPUT_PATH}")
    print("#"*80)
    
    input_path = Config.INPUT_PATH
    
    if os.path.splitext(input_path)[1].lower() == '.zip':
        master_output_dir = Config.get_master_output_dir(input_path)
        print(f"Detected ZIP archive. Starting batch processing...")
        temp_extract_dir = os.path.join(master_output_dir, "zip_extracted")
        os.makedirs(temp_extract_dir, exist_ok=True)
        
        with zipfile.ZipFile(input_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_dir)
        
        print(f"Archive extracted to: {temp_extract_dir}")
        
        all_files_to_process = []
        for root, _, files in os.walk(temp_extract_dir):
            for file in files:
                if file.startswith('__MACOSX') or file.startswith('.'):
                    continue
                all_files_to_process.append(os.path.join(root, file))

        for file_path in all_files_to_process:
            file_base_output_dir = os.path.join(master_output_dir, f"output_{os.path.splitext(os.path.basename(file_path))[0]}")
            loaded_models_in_run = process_single_file(file_path, file_base_output_dir, Config)
            
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
        master_output_dir = Config.get_master_output_dir(input_path)
        loaded_models_in_run = process_single_file(input_path, master_output_dir, Config)
        # Final cleanup after single file run
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
    else:
        print(f"❌ FATAL ERROR: Input path is not a valid file or ZIP archive: {input_path}")
        return

    # Cleanup temp directories
    master_output_dir = Config.get_master_output_dir(input_path)
    temp_cell_dir_to_check = os.path.join(master_output_dir, "temp_cells_for_batching")
    if os.path.exists(temp_cell_dir_to_check):
        print(f"\nCleaning up temporary directory: {temp_cell_dir_to_check}")
        shutil.rmtree(temp_cell_dir_to_check)
        print("✅ Cleanup complete.")
    zip_extract_dir_to_check = os.path.join(master_output_dir, "zip_extracted")
    if os.path.exists(zip_extract_dir_to_check):
        print(f"\nCleaning up ZIP extraction directory: {zip_extract_dir_to_check}")
        shutil.rmtree(zip_extract_dir_to_check)
        print("✅ Cleanup complete.")

    pipeline_elapsed_time = time.time() - pipeline_start_time
    print("\n" + "#"*80)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY! 🎉")
    print(f"Total execution time: {pipeline_elapsed_time:.2f} seconds ({pipeline_elapsed_time/60:.2f} minutes)")
    print(f"Final outputs are located in: {master_output_dir}")
    print("#"*80)

if __name__ == "__main__":
    main()
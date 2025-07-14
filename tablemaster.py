# -*- coding: utf-8 -*-
# ======================================================================================
# 全文档处理流水线 (Complete Document Processing Pipeline) - 专家增强版 V7
#
# 本脚本由专家团队审查和重构，整合了从PDF到最终Markdown文档的完整处理流程，
# 并引入了极致灵活的模型选择机制和针对gpt-4o及本地模型的高性能批量处理功能。
#
# 专家增强版 V7 更新说明 (Expert Enhanced V7 Update Notes):
# - 本地模型批量处理 (Batch Processing for Local Models): 为本地部署的Qwen和Nanonets模型
#   实现了高效的批量处理（Batch Processing）功能。现在，对于有框线表格的单元格识别（Qwen）
#   和无框线表格的整表识别（Nanonets），模型将一次性处理一批图片，大幅减少模型调用开销，
#   显著提升处理速度，特别是对于包含大量表格的文档。
# - VLM_BATCH_SIZE 配置: 在`Config`类中新增`VLM_BATCH_SIZE`参数，允许您根据
#   硬件（如VRAM大小）灵活调整本地模型处理的批量大小，以实现最佳性能。
# - 鲁棒的失败重试 (Robust Failure Retry): 为本地模型的批量处理增加了失败重试机制。
#   如果批处理中的某个项目失败，系统会自动切换到单项处理模式对该项目进行重试，
#   确保了整个流程的健壮性。
# - 极致模型选择 (Ultimate Model Selector): 您现在可以通过修改`Config.ModelSelector`类，为
#   流水线中的每一个关键AI任务（图像描述、无框线/有框线表格识别、表格合并、标题分析）
#   独立选择使用本地部署的VLM（Qwen, Nanonets）或OpenAI的gpt-4o/GPT-4.1-mini模型。
# - gpt-4o并发批量处理 (Concurrent Batch Processing for gpt-4o): 为需要处理大量小图的
#   任务（如图像描述、表格单元格识别），实现了基于线程池的并发API请求，
#   极大提升处理效率。
# - 专家级代码审查与注释 (Expert Code Review & Comments): 全面审查代码逻辑，
#   添加了详尽的注释，使工作流和复杂功能更易于理解和维护。
#
# 流程 (Pipeline Flow):
# 1. PDF转图片 -> 2. 版面分析 -> 3. 视觉元素裁剪 (表格+图像) ->
# 4. 识别引擎 (模型可完全自定义):
#    4a. 图像内容识别 (local_qwen / gpt-4o)
#    4b. 表格识别 (local_nanonets / gpt-4o / local_qwen) - 【V7 批量处理增强】
# 5. AI表格合并 (gpt-4.1-mini / gpt-4o) -> 6. 结果聚合 -> 7. 标题分级与Markdown生成
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
from PIL import Image
import base64
import requests
import shutil
from pdf2image import convert_from_path
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ======================================================================================
# --- ⚙️ 统一配置 (Unified Configuration) ---
# ======================================================================================
class Config:
    """
    集中管理所有路径、模型和API密钥的配置类。
    Configuration class to centrally manage all paths, models, and API keys.
    """
    # --- 输入文件 (Input File) ---
    PDF_PATH = "/project/chenjian/bbb/[定期报告][2023-03-20][朗鸿科技]朗鸿科技2022年年度报告摘要.pdf"

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
        # 任务1: 图像描述 (图表、照片等)
        IMAGE_DESCRIPTION = 'gpt-4o' # 可选: 'local_qwen', 'gpt-4o'

        # 任务2: 无框线表格识别
        BORDERLESS_TABLE_RECOGNITION = 'gpt-4o' # 可选: 'local_nanonets', 'gpt-4o'

        # 任务3: 有框线表格的单元格内容识别
        BORDERED_TABLE_CELL_RECOGNITION = 'gpt-4o' # 可选: 'local_qwen', 'gpt-4o'

        # 任务4: 跨页表格的智能合并 (需要强大的逻辑推理能力)
        TABLE_MERGING = 'gpt-4.1-mini-2025-04-14' # 可选: 'gpt-4o', 'gpt-4.1-mini-2025-04-14'

        # 任务5: 文档标题层级分析 (需要强大的文档结构理解能力)
        TITLE_HIERARCHY = 'gpt-4.1-mini-2025-04-14' # 可选: 'gpt-4o', 'gpt-4.1-mini-2025-04-14'

    # --- 处理参数 (Processing Parameters) ---
    PDF_TO_IMAGE_DPI = 200
    API_REQUEST_TIMEOUT = 120 # API请求超时时间（秒）
    GPT4O_BATCH_SIZE = 100     # gpt-4o 并发请求的数量
    VLM_BATCH_SIZE = 16       # 【V7 新增】本地VLM模型批处理大小 (根据VRAM调整)

    # --- 动态生成的输出目录 (Dynamically Generated Output Directories) ---
    if not os.path.exists(PDF_PATH):
        print(f"❌ 致命错误: 配置文件中的PDF路径不存在: {PDF_PATH}")
        sys.exit(1)

    MASTER_OUTPUT_DIR = os.path.join(os.path.dirname(PDF_PATH), f"output_{os.path.splitext(os.path.basename(PDF_PATH))[0]}")

    # 各个阶段的子目录
    DIR_1_PAGE_IMAGES = os.path.join(MASTER_OUTPUT_DIR, "1_page_images")
    DIR_2_LAYOUT_JSONS = os.path.join(MASTER_OUTPUT_DIR, "2_layout_jsons")
    DIR_3_CROPPED_TABLES = os.path.join(MASTER_OUTPUT_DIR, "3_cropped_tables")
    DIR_3_CROPPED_IMAGES = os.path.join(MASTER_OUTPUT_DIR, "3_cropped_images")
    DIR_TEMP_CELLS = os.path.join(MASTER_OUTPUT_DIR, "temp_cells_for_batching")

    # --- 最终输出文件路径 (Final Output File Paths) ---
    FINAL_COMBINED_JSON_PATH = os.path.join(MASTER_OUTPUT_DIR, "_combined_document.json")
    FINAL_JSON_WITH_HIERARCHY_PATH = os.path.join(MASTER_OUTPUT_DIR, "_document_with_hierarchy.json")
    FINAL_MARKDOWN_FILENAME_PATH = os.path.join(MASTER_OUTPUT_DIR, "_final_document.md")

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
# --- STAGE 2: Layout Analysis ---
# ======================================================================================
def run_step_2_layout_analysis(input_dir, output_dir):
    """Performs layout analysis on images using PaddleX and saves results as JSON."""
    print("\n" + "="*80 + "\n--- STAGE 2: Layout Analysis ---\n" + "="*80)
    os.makedirs(output_dir, exist_ok=True)
    from paddlex import create_pipeline as pp_create_pipeline
    pipeline = None
    try:
        print("Initializing PP-StructureV3 pipeline...")
        pipeline = pp_create_pipeline(pipeline="layout_parsing")
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

# ======================================================================================
# --- STAGE 3: Visual Element Cropping (Tables & Images) ---
# ======================================================================================
def run_step_3_crop_visual_elements(image_dir, json_dir, table_output_dir, image_output_dir):
    """Crops table and image regions from original page images based on layout analysis."""
    print("\n" + "="*80 + "\n--- STAGE 3: Visual Element Cropping (Tables & Images) ---\n" + "="*80)
    os.makedirs(table_output_dir, exist_ok=True)
    os.makedirs(image_output_dir, exist_ok=True)
    total_tables_found, total_images_found = 0, 0
    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])
    if not json_files:
        print("⚠️ WARNING: No JSON layout files found in Stage 3. Skipping.")
        return True

    start_time = time.time()
    for json_filename in tqdm(json_files, desc="Cropping Tables & Images"):
        base_filename = os.path.splitext(json_filename)[0]
        json_path = os.path.join(json_dir, json_filename)
        image_path = os.path.join(image_dir, f"{base_filename}.png")
        if not os.path.exists(image_path):
            tqdm.write(f" [SKIP] Image not found for JSON file: {image_path}")
            continue
        try:
            image = Image.open(image_path)
            with open(json_path, 'r', encoding='utf-8') as f:
                layout_data = json.load(f)
            if "parsing_res_list" not in layout_data:
                continue
            for i, block in enumerate(layout_data["parsing_res_list"]):
                block_label = block.get("block_label")
                bbox = block.get("block_bbox")
                if not bbox: continue
                x1, y1, x2, y2 = map(int, bbox)
                if x1 >= x2 or y1 >= y2: continue
                cropped_image = image.crop((max(0, x1-5), max(0, y1-10), x2+5, y2+5))
                if block_label == "table":
                    output_filename = f"{base_filename}_table_{i}.jpg"
                    output_path = os.path.join(table_output_dir, output_filename)
                    cropped_image.save(output_path)
                    total_tables_found += 1
                elif block_label == "image":
                    output_filename = f"{base_filename}_image_{i}.jpg"
                    output_path = os.path.join(image_output_dir, output_filename)
                    cropped_image.save(output_path)
                    total_images_found += 1

        except Exception as e:
            tqdm.write(f" [ERROR] Failed to process {json_filename}: {e}")
    elapsed_time = time.time() - start_time
    print(f"✅ STAGE 3 Complete: Cropped {total_tables_found} tables and {total_images_found} images in {elapsed_time:.2f} seconds.")
    print(f" ➡️ Tables saved to: {table_output_dir}")
    print(f" ➡️ Images saved to: {image_output_dir}")
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
def batch_recognize_text_with_qwen(image_batch, model, processor):
    """【V7】Recognizes text from a BATCH of images using the Qwen-VL model."""
    if not image_batch: return []
    try:
        pil_images = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in image_batch]
        prompt = "直接提取图片中的所有文字内容。注意如果是空白的图片的话返回 ''。"
        messages_batch = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}] for _ in pil_images]
        text_batch = [processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_batch]
        
        inputs = processor(text=text_batch, images=pil_images, padding=True, return_tensors="pt").to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
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
                html_content += f" <td{rowspan_attr}{colspan_attr}>{cell_text}</td>\n"
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
        os.makedirs(config.DIR_TEMP_CELLS, exist_ok=True)
        for table_path in tqdm(bordered_tables_for_gpt, desc="Slicing Bordered Tables (for gpt-4o)"):
            table_filename = os.path.basename(table_path)
            original_img_np = cv2.imread(table_path)
            cells_with_coords = intelligent_cell_slicer(original_img_np)
            if not cells_with_coords:
                table_html_results[table_filename] = "<table><tr><td>Error: Cell slicing failed.</td></tr></table>"
                continue
            
            cell_paths = []
            for i, (cell_img, _) in enumerate(cells_with_coords):
                path = os.path.join(config.DIR_TEMP_CELLS, f"{os.path.splitext(table_filename)[0]}_cell_{i}.png")
                cv2.imwrite(path, cell_img)
                cell_paths.append(path)
            
            # prompt_cell = "直接提取图片中的所有文字内容。如果是空白的图片，请返回''。只输出识别到的内容。"
            # cell_ocr_results = batch_recognize_with_openai_vision(client, 'gpt-4o', cell_paths, prompt_cell, config)
            # cell_texts = ["" if (text := cell_ocr_results.get(os.path.basename(p), "")) == "''" else text for p in cell_paths]
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

            # --- 第二部分：定义一个用于解析的函数 ---

            def extract_content_from_template(text: str) -> str:
                """
                使用正则表达式从模板中提取内容。
                """
                # re.escape() 确保标记中的特殊字符不会影响正则匹配
                # (.*?) 是一个非贪婪匹配，匹配开始和结束标记之间的所有字符，包括换行符
                # re.DOTALL 标志让 . 可以匹配包括换行符在内的任意字符
                pattern = re.compile(f"{re.escape(START_MARKER)}(.*?){re.escape(END_MARKER)}", re.DOTALL)
                
                match = pattern.search(text)
                
                if match:
                    # group(1) 获取第一个捕获组（也就是括号里的内容）
                    # .strip() 用于移除可能存在的前后空白或换行符
                    return match.group(1).strip()
                else:
                    # 如果在返回结果中找不到模板，可以选择返回空字符串或原始文本
                    # 返回空字符串通常更安全
                    return ""

            # --- 第三部分：执行并解析结果 ---

            # 假设 client, cell_paths, config 已经定义好了
            # client = ...
            # cell_paths = [...]
            # config = ...

            # 1. 使用新的 prompt 调用 OCR 服务
            cell_ocr_results = batch_recognize_with_openai_vision(client, 'gpt-4o', cell_paths, prompt_cell, config)

            # 2. 使用新的解析函数处理结果
            # .get(os.path.basename(p), "") 确保即使某个文件识别失败，也能得到一个空字符串，避免程序出错
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

# ======================================================================================
# --- STAGE 5: AI-Powered Intelligent Table Merging ---
# ======================================================================================
def run_step_5_ai_merge_tables(final_results_dir, client, config):
    """
    Identifies and merges STRICTLY ADJACENT tables using a pairwise, rolling AI-driven approach.
    
    This function implements a highly precise "rolling merge" strategy on adjacent blocks:
    1. It loads all document blocks from all pages into a single, ordered list.
    2. It iterates through this list, specifically looking for a block labeled 'table'
       that is immediately followed by another block also labeled 'table'.
    3. Only for these adjacent pairs does it ask an AI model if the second table is a logical 
       continuation of the first.
    4. IF THEY MERGE:
       - The two tables are combined into a single, revised HTML table by the AI.
       - This new, merged table replaces the first table block.
       - The second table block is marked as 'merged_into_previous'.
       - The process then continues, comparing the newly formed table with the *next* block
         to see if it's also a table that can be merged in.
    5. IF THEY DO NOT MERGE (or are not adjacent):
       - The blocks are left as they are.
       - The process moves on to the next block to check for a new adjacent pair.
    """
    print("\n" + "="*80 + "\n--- STAGE 5: AI-Powered Adjacent Table Merging (Expert Logic v2) ---\n" + "="*80)
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
    # This is critical for checking for *strictly adjacent* tables.
    all_block_refs = []
    for p_idx, page_data in enumerate(all_pages_data):
        for b_idx, block in enumerate(page_data.get('parsing_res_list', [])):
            all_block_refs.append({
                # Add a direct reference to the block for easier modification
                'block_data': block,
                # Keep original location info for logging
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
                decision = run_ai_analysis(client, model_name, decision_prompt)
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
                    revised_content = run_ai_analysis(client, model_name, revision_prompt)
                except Exception as e:
                    print(f"ERROR: An exception occurred during AI revision: {e}")
                    revised_content = None

                if revised_content and "revised_html" in revised_content and revised_content["revised_html"]:
                    print(" -> AI revision successful. Updating document structure.")
                    
                    current_block['block_content'] = revised_content["revised_html"]
                    next_block['block_label'] = 'merged_into_previous'
                    next_block['block_content'] = ''
                    
                    # Remove the now-merged block reference from our list
                    all_block_refs.pop(i + 1)
                    
                    # DO NOT increment 'i'. The newly merged table at index 'i' now needs to be
                    # compared with its new neighbor (the original block at i+2).
                    continue
                else:
                    print(" -> AI revision failed or returned empty content. Not merging this pair.")
            else:
                print(" -> AI Decision: DO NOT MERGE or analysis failed. Treating as separate tables.")
        
        # If not an adjacent table pair, or if merge failed/declined, move to the next block.
        i += 1

    # --- 6. Save all modified data back to JSON files ---
    print("\nSaving all updated files...")
    for p_idx, page_data in enumerate(tqdm(all_pages_data, desc="Saving merged files")):
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
# --- 🚀 Main Execution Orchestrator ---
# ======================================================================================
def main():
    """Main function to execute the entire document processing pipeline in order."""
    pipeline_start_time = time.time()
    print("#"*80)
    print("🚀 STARTING DOCUMENT PROCESSING PIPELINE (ENHANCED V5) 🚀")
    print(f"🕒 Start Time: {time.ctime(pipeline_start_time)}")
    print(f"📁 Master Output Directory: {Config.MASTER_OUTPUT_DIR}")
    print("#"*80)
    
    # --- STAGE 1 ---
    if not run_step_1_pdf_to_images(Config.PDF_PATH, Config.DIR_1_PAGE_IMAGES, Config.PDF_TO_IMAGE_DPI):
        print("Pipeline halted at Stage 1."); return

    # --- STAGE 2 ---
    if not run_step_2_layout_analysis(Config.DIR_1_PAGE_IMAGES, Config.DIR_2_LAYOUT_JSONS):
        print("Pipeline halted at Stage 2."); return

    # --- STAGE 3 ---
    if not run_step_3_crop_visual_elements(Config.DIR_1_PAGE_IMAGES, Config.DIR_2_LAYOUT_JSONS, Config.DIR_3_CROPPED_TABLES, Config.DIR_3_CROPPED_IMAGES):
        print("Pipeline halted at Stage 3."); return

    # --- Dynamic Model & Client Loading ---
    print("\n" + "-"*25 + " Initializing Models & Clients " + "-"*25)
    local_models = initialize_local_models(Config)
    openai_client = initialize_openai_client(Config)
    clients_and_models = {'openai': openai_client, **local_models}
    print("-" * 80)
    
    # --- STAGE 4a ---
    if not run_step_4a_recognize_images(Config.DIR_3_CROPPED_IMAGES, Config.DIR_2_LAYOUT_JSONS, clients_and_models, Config):
        print("Pipeline halted at Stage 4a."); return

    # --- STAGE 4b ---
    if not run_step_4b_process_all_tables(Config.DIR_3_CROPPED_TABLES, Config.DIR_2_LAYOUT_JSONS, clients_and_models, Config):
        print("Pipeline halted at Stage 4b."); return
        
    # --- Release Local Models (if loaded) ---
    print("\nReleasing local VLM models from memory (if they were loaded)...")
    if local_models.get('qwen'): del local_models['qwen']
    if local_models.get('nanonets'): del local_models['nanonets']
    del local_models
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("✅ PyTorch CUDA cache cleared.")
    except (ImportError, NameError): pass
    print("✅ Local model resources released.")

    # --- STAGE 5 ---
    if not run_step_5_ai_merge_tables(Config.DIR_2_LAYOUT_JSONS, openai_client, Config):
        print("Pipeline halted at Stage 5."); return

    # --- STAGE 6 ---
    if not run_step_6_aggregate_results(Config.DIR_2_LAYOUT_JSONS, Config.FINAL_COMBINED_JSON_PATH):
        print("Pipeline halted at Stage 6."); return

    # --- STAGE 7 ---
    if not run_step_7_create_final_document(
        Config.FINAL_COMBINED_JSON_PATH,
        Config.FINAL_JSON_WITH_HIERARCHY_PATH,
        Config.FINAL_MARKDOWN_FILENAME_PATH,
        Config.API_KEY,
        Config.API_BASE_URL,
        Config.ModelSelector.TITLE_HIERARCHY
    ):
        print("Pipeline halted at Stage 7."); return
        
    # --- Final Cleanup ---
    if os.path.exists(Config.DIR_TEMP_CELLS):
        print(f"\nCleaning up temporary directory: {Config.DIR_TEMP_CELLS}")
        shutil.rmtree(Config.DIR_TEMP_CELLS)
        print("✅ Cleanup complete.")

    # --- Final Summary ---
    pipeline_elapsed_time = time.time() - pipeline_start_time
    print("\n" + "#"*80)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY! 🎉")
    print(f"Total execution time: {pipeline_elapsed_time:.2f} seconds ({pipeline_elapsed_time/60:.2f} minutes)")
    print(f"Final outputs are located in: {Config.MASTER_OUTPUT_DIR}")
    print("#"*80)

if __name__ == "__main__":
    # Create all necessary directories before starting
    os.makedirs(Config.MASTER_OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.DIR_1_PAGE_IMAGES, exist_ok=True)
    os.makedirs(Config.DIR_2_LAYOUT_JSONS, exist_ok=True)
    os.makedirs(Config.DIR_3_CROPPED_TABLES, exist_ok=True)
    os.makedirs(Config.DIR_3_CROPPED_IMAGES, exist_ok=True)
    main()

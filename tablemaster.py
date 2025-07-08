# -*- coding: utf-8 -*-
"""
PDF Table Extractor and HTML Converter with Advanced Cell Recognition

This script provides an end-to-end pipeline to extract tables from a PDF file
and convert them into HTML format. It integrates several main stages:
1. PDF to Image Conversion.
2. Layout Analysis with PaddleOCR to find table locations.
3. Cropping of table areas.
4. Advanced Table Structure Recognition:
   a. Intelligently slices tables into individual cells using a hybrid strategy.
   b. Uses a Vision Language Model (VLM) for OCR on each cell.
   c. Reconstructs a full HTML table, calculating rowspans and colspans.
5. AI-powered merging of tables that span across pages.
6. Final aggregation of all content into a structured JSON file.

The entire process is automated. You only need to set the configuration parameters.
"""
import os
import shutil
import json
import re
import time
import gc
from multiprocessing import Pool, set_start_method
from openai import OpenAI
from bs4 import BeautifulSoup
import html
import traceback

# --- Third-party library imports ---
# Make sure you have installed all required libraries:
# pip install torch transformers pillow pdf2image paddlex "paddlenlp>=2.5" tqdm sentencepiece accelerate opencv-python numpy pynvml beautifulsoup4 lxml

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
from pdf2image import convert_from_path
from paddlex import create_pipeline
import paddle
from tqdm import tqdm
from pynvml import *
import cv2
import numpy as np


# ======================================================================================
# --- GPU Utility ---
# ======================================================================================
def get_available_gpus(required_free_gb=10):
    """
    Finds GPUs that have at least a certain amount of free memory.
    """
    available_gpus = []
    try:
        nvmlInit()
        device_count = nvmlDeviceGetCount()
        print(f"🔍 Found {device_count} NVIDIA GPUs. Checking their status...")
        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            free_gb = mem_info.free / (1024**3)
            if free_gb >= required_free_gb:
                print(f" ✅ GPU {i}: Available (Free Memory: {free_gb:.2f} GB)")
                available_gpus.append(i)
            else:
                print(f" ❌ GPU {i}: Occupied (Free Memory: {free_gb:.2f} GB) - Skipping.")
        nvmlShutdown()
    except NVMLError as error:
        print(f"Error while querying GPUs with NVML: {error}. Will proceed without dynamic selection.")
        # Fallback to all visible devices if NVML fails
        return list(range(torch.cuda.device_count()))
    return available_gpus


# --- ⚙️ Main Configuration ---
class Config:
    # --- Paths and Models ---
    PDF_PATH = "/project/chenjian/data cleansing/[定期报告][2023-03-20][朗鸿科技]朗鸿科技2022年年度报告摘要.pdf"
    # Use the Qwen-VL model as specified in the new logic
    VLM_MODEL_CHECKPOINT = "/project/chenjian/Qwen/Qwen2.5-VL-7B-Instruct"
    # --- Performance ---
    BATCH_SIZE_PER_GPU = 1 # Kept at 1 for large models
    REQUIRED_FREE_MEM_GB = 20 # Memory requirement for the Qwen-VL model
    PDF_TO_IMAGE_DPI = 300

    # --- AI Post-processing (for Stage 4.5) ---
    API_KEY = "sk-3ni5O4wR7GTeeqKvFdC5D12f280b460797E7369455283a7d"
    API_BASE_URL = "http://152.53.52.170:3003/v1"
    AI_MODEL_NAME = "gpt-4.1-mini-2025-04-14"

# ======================================================================================
# --- STAGE 1: PDF to Image Conversion ---
# ======================================================================================
def convert_pdf_to_images(pdf_path, output_dir):
    """
    Converts each page of a PDF file into a separate PNG image.
    """
    print("\n" + "="*80 + "\n--- STAGE 1: Starting PDF to Image Conversion ---\n" + "="*80)
    if not os.path.exists(pdf_path):
        print(f"❌ ERROR: PDF file not found at '{pdf_path}'. Please check the path in Config.")
        return False

    os.makedirs(output_dir, exist_ok=True)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    try:
        start_time = time.time()
        images = convert_from_path(pdf_path, dpi=Config.PDF_TO_IMAGE_DPI)
        for i, image in enumerate(tqdm(images, desc="Converting PDF pages")):
            output_path = os.path.join(output_dir, f"{pdf_name}_page_{i+1}.png")
            image.save(output_path, 'PNG')
        elapsed_time = time.time() - start_time
        print(f"✅ STAGE 1 Complete: Successfully converted {len(images)} pages in {elapsed_time:.2f} seconds.")
        return True
    except Exception as e:
        print(f"❌ ERROR in Stage 1: PDF to image conversion failed. Details: {e}")
        return False

# ======================================================================================
# --- STAGE 2: Layout Analysis with PaddleOCR ---
# ======================================================================================
def analyze_document_layout(image_dir, output_dir):
    """
    Analyzes the layout of each page image to identify tables using PaddleOCR.
    """
    print("\n" + "="*80 + "\n--- STAGE 2: Starting Document Layout Analysis ---\n" + "="*80)
    os.makedirs(output_dir, exist_ok=True)

    try:
        print("Initializing PP-StructureV3 pipeline...")
        # FIX: Use GPU for PaddleOCR if available for much faster processing.
        if paddle.is_compiled_with_cuda():
            print("✅ PaddleOCR: GPU detected. Setting device to 'gpu' for faster layout analysis.")
            paddle.set_device('gpu')
        else:
            print("⚠️ PaddleOCR: No GPU detected. Setting device to 'cpu'. Layout analysis will be slower.")
            paddle.set_device('cpu')
            
        pipeline = create_pipeline(pipeline="PP-StructureV3")
        print("Pipeline initialized.")

        image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not image_files:
            print("❌ ERROR: No images found in the input directory for Stage 2.")
            return False

        start_time = time.time()
        for filename in tqdm(image_files, desc="Analyzing page layouts"):
            input_path = os.path.join(image_dir, filename)
            # Run layout analysis
            output = pipeline.predict(
                input=input_path,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
            base_filename = os.path.splitext(filename)[0]
            json_save_path = os.path.join(output_dir, f"{base_filename}.json")
            if output:
                # Save the first result to JSON
                for res in output:
                    res.save_to_json(save_path=json_save_path)
                    break

        elapsed_time = time.time() - start_time
        print(f"✅ STAGE 2 Complete: Analyzed {len(image_files)} images in {elapsed_time:.2f} seconds.")
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"❌ ERROR in Stage 2: Layout analysis failed. Details: {e}\n{traceback.format_exc()}")
        return False

# ======================================================================================
# --- STAGE 3: Cropping Tables ---
# ======================================================================================
def crop_tables_from_images(image_dir, json_dir, output_dir):
    """
    Crops table areas from the original page images based on layout analysis results.
    """
    print("\n" + "="*80 + "\n--- STAGE 3: Table Cropping ---\n" + "="*80)
    os.makedirs(output_dir, exist_ok=True)
    total_tables_found = 0
    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])
    if not json_files:
        print("⚠️ WARNING: No JSON layout files found. Skipping Stage 3.")
        return 0

    start_time = time.time()
    for json_filename in tqdm(json_files, desc="Cropping Tables"):
        base_filename = os.path.splitext(json_filename)[0]
        json_path = os.path.join(json_dir, json_filename)
        image_path = os.path.join(image_dir, f"{base_filename}.png")
        if not os.path.exists(image_path):
            print(f" [SKIP] Image not found for JSON file: {image_path}")
            continue

        try:
            image = Image.open(image_path)
            with open(json_path, 'r', encoding='utf-8') as f:
                layout_data = json.load(f)

            if "parsing_res_list" not in layout_data:
                continue

            for i, block in enumerate(layout_data["parsing_res_list"]):
                if block.get("block_label") == "table":
                    bbox = block.get("block_bbox")
                    if not bbox: continue
                    # Add a small margin around the table
                    x1, y1, x2, y2 = map(int, bbox)
                    if x1 >= x2 or y1 >= y2: continue

                    cropped_table_image = image.crop((x1-5, y1-10, x2+5, y2+5))
                    output_filename = f"{base_filename}_table_{i}.jpg"
                    output_path = os.path.join(output_dir, output_filename)
                    cropped_table_image.save(output_path)
                    total_tables_found += 1
        except Exception as e:
            print(f" [ERROR] Failed to process {json_filename}: {e}")
    elapsed_time = time.time() - start_time
    print(f"✅ STAGE 3 Complete: Found and cropped {total_tables_found} tables in {elapsed_time:.2f} seconds.")
    return total_tables_found


# ======================================================================================
# --- STAGE 4: ADVANCED TABLE TO HTML CONVERSION (NEW INTEGRATED LOGIC) ---
# This section contains the new, advanced table processing pipeline.
# ======================================================================================

# --- Module 4.1: VLM Model and Helpers ---
def process_vision_info(messages):
    """Helper function to extract image/video data for the VLM processor."""
    image_inputs = []
    video_inputs = []
    if messages and messages[0]['role'] == 'user':
        for content in messages[0].get('content', []):
            if content.get('type') == 'image':
                image_inputs.append(content['image'])
            elif content.get('type') == 'video':
                video_inputs.append(content['video'])
    return image_inputs, video_inputs

def initialize_vlm():
    """Initializes and returns the Qwen-VL model and processor."""
    print("Loading Qwen2.5-VL-7B-Instruct model and processor...")
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            Config.VLM_MODEL_CHECKPOINT,
            torch_dtype="auto",
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(Config.VLM_MODEL_CHECKPOINT)
        print("✅ Qwen-VL model and processor loaded successfully!")
        return model, processor
    except Exception as e:
        print(f"❌ VLM model loading failed: {e}")
        return None, None

def recognize_text_with_vlm(image_np, model, processor):
    """Recognizes text in a single image using the Qwen-VL model."""
    if image_np.size == 0: return "", 0.0
    try:
        image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        prompt = "What is the text in this image? Return only the text content. If the image is blank, return Null."
        messages = [{"role": "user", "content": [{"type": "image", "image": image_pil}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        cleaned_text = output_text.strip().strip('"').strip("'")
        return cleaned_text, 1.0
    except Exception as e:
        print(f" (Warning: VLM recognition failed: {e})")
        return "", 0.0

# --- Module 4.2: Intelligent Cell Slicer ---
def _cut_cells_by_finding_contours(img, height, width):
    """Strategy A: Based on finding contours. Suitable for closed-border tables."""
    print("--- Trying Strategy A: Contour-based (for closed tables) ---")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    divisor, iterations = 40, 1
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, width // divisor), 1))
    detected_horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=iterations)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, height // divisor)))
    detected_vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=iterations)
    table_grid_mask = cv2.add(detected_horizontal, detected_vertical)
    cells_mask = 255 - table_grid_mask
    contours, _ = cv2.findContours(cells_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Strategy A: Found {len(contours)} potential contours.")
    if len(contours) <= 1:
        print("Strategy A: Too few contours found. This might be an open table. Strategy A is not suitable.")
        return []
    contours = sorted(contours, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))
    results = []
    for contour in contours:
        if cv2.contourArea(contour) < 100: continue
        x, y, w, h = cv2.boundingRect(contour)
        cell_image = img[y:y+h, x:x+w]
        if cell_image.size > 0:
            results.append((cell_image, (x, y, w, h)))
    return results

def _cut_cells_by_line_coordinates(img, height, width):
    """Strategy B: Based on locating line coordinates. Suitable for open tables."""
    print("--- Switching to Strategy B: Coordinate-based (for open tables) ---")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    divisor, iterations = 15, 1
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, width // divisor), 1))
    detected_horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=iterations)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, height // divisor)))
    detected_vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=iterations)
    horizontal_contours, _ = cv2.findContours(detected_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vertical_contours, _ = cv2.findContours(detected_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    row_boundaries, col_boundaries = set(), set()
    for contour in horizontal_contours:
        x, y, w, h = cv2.boundingRect(contour)
        row_boundaries.add(y); row_boundaries.add(y+h)
    for contour in vertical_contours:
        x, y, w, h = cv2.boundingRect(contour)
        col_boundaries.add(x); col_boundaries.add(x+w)
    def merge_close_boundaries(boundaries, tolerance=5):
        if not boundaries: return []
        sorted_boundaries = sorted(list(boundaries))
        merged = [sorted_boundaries[0]]
        for b in sorted_boundaries[1:]:
            if b - merged[-1] > tolerance: merged.append(b)
        return merged
    final_row_boundaries = merge_close_boundaries(row_boundaries)
    final_col_boundaries = merge_close_boundaries(col_boundaries)
    if len(final_row_boundaries) < 2 or len(final_col_boundaries) < 2:
        print("Strategy B: Not enough row or column boundaries found.")
        return []
    results = []
    for r in range(len(final_row_boundaries) - 1):
        for c in range(len(final_col_boundaries) - 1):
            y_start, y_end = final_row_boundaries[r], final_row_boundaries[r+1]
            x_start, x_end = final_col_boundaries[c], final_col_boundaries[c+1]
            cell_image = img[y_start:y_end, x_start:x_end]
            if cell_image.size > 0:
                results.append((cell_image, (x_start, y_start, x_end - x_start, y_end - y_start)))
    return results

def intelligent_cell_slicer(img):
    """Intelligent slicer main function using a hybrid strategy."""
    if img is None:
        print("Error: Input image to slicer is None.")
        return []
    height, width, _ = img.shape
    cells = _cut_cells_by_finding_contours(img, height, width)
    if not cells:
        print("Strategy A failed, switching to Strategy B.")
        cells = _cut_cells_by_line_coordinates(img, height, width)
    if not cells:
        print("WARNING: Both slicing strategies failed. Treating the entire image as a single cell.")
        cells = [(img, (0, 0, width, height))]
    return cells

# --- Module 4.3: Cell Data Extraction ---
def extract_cell_data(image_path, vlm_model, vlm_processor):
    """Uses hybrid strategy to slice cells and VLM for OCR, returning structured data."""
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}"); return None
    original_img = cv2.imread(image_path)
    if original_img is None:
        print("Error: Could not read image."); return None
    cells_with_coords = intelligent_cell_slicer(original_img)
    if not cells_with_coords:
        print(f"❌ WARNING: No cells found in {os.path.basename(image_path)}.")
        return None
    table_data = []
    total_cells = len(cells_with_coords)
    print(f"\n✅ Slicing complete, found {total_cells} cells. Starting VLM recognition...")
    for i, (cell_image, (x, y, w, h)) in enumerate(cells_with_coords):
        padding = 1
        padded_cell_image = cell_image[padding:h-padding, padding:w-padding]
        if padded_cell_image.size == 0: continue
        text, score = recognize_text_with_vlm(padded_cell_image, vlm_model, vlm_processor)
        cell_info = {
            "coordinates": {"x": x, "y": y, "width": w, "height": h},
            "recognition_result": {"text": text, "score": score}
        }
        table_data.append(cell_info)
        print(f"  Processed cell {i+1}/{total_cells}: text='{text}'")
    return table_data

# --- Module 4.4: HTML Reconstruction with Spans ---
def find_closest_index(boundary_list, value, tolerance=5):
    """Finds the closest index for a value in a boundary list."""
    for i, boundary in enumerate(boundary_list):
        if abs(value - boundary) < tolerance:
            return i
    return -1

def convert_data_to_html_with_spans(cells_data):
    """Reads cell data, reconstructs a table with merged cells, and returns an HTML string."""
    if not cells_data:
        print("Warning: Cell data is empty, cannot generate HTML.")
        return ""
    x_boundaries, y_boundaries = set(), set()
    for cell in cells_data:
        coords = cell['coordinates']
        x_boundaries.add(coords['x']); x_boundaries.add(coords['x'] + coords['width'])
        y_boundaries.add(coords['y']); y_boundaries.add(coords['y'] + coords['height'])
    def merge_boundaries(boundaries, tolerance=5):
        if not boundaries:
            return []
        # FIX: Convert the set to a sorted list before accessing elements by index
        sorted_boundaries = sorted(list(boundaries))
        merged = [sorted_boundaries[0]]
        for b in sorted_boundaries[1:]:
            if b - merged[-1] > tolerance:
                merged.append(b)
        return merged
    final_x_boundaries = merge_boundaries(x_boundaries)
    final_y_boundaries = merge_boundaries(y_boundaries)
    num_cols = len(final_x_boundaries) - 1
    num_rows = len(final_y_boundaries) - 1
    if num_cols <= 0 or num_rows <= 0:
        print("Error: Could not determine table grid structure.")
        return ""
    grid = [[None for _ in range(num_cols)] for _ in range(num_rows)]
    for cell in cells_data:
        coords = cell['coordinates']
        start_col = find_closest_index(final_x_boundaries, coords['x'])
        end_col = find_closest_index(final_x_boundaries, coords['x'] + coords['width'])
        start_row = find_closest_index(final_y_boundaries, coords['y'])
        end_row = find_closest_index(final_y_boundaries, coords['y'] + coords['height'])
        if -1 in [start_col, end_col, start_row, end_row]: continue
        colspan = end_col - start_col
        rowspan = end_row - start_row
        if start_row < num_rows and start_col < num_cols:
            grid[start_row][start_col] = {
                'text': cell['recognition_result']['text'],
                'rowspan': rowspan, 'colspan': colspan, 'is_placed': False
            }
    html_content = "<table>\n"
    for r in range(num_rows):
        html_content += "  <tr>\n"
        for c in range(num_cols):
            cell = grid[r][c]
            if cell:
                if cell['is_placed']: continue
                for i in range(cell['rowspan']):
                    for j in range(cell['colspan']):
                        if (r + i) < num_rows and (c + j) < num_cols and grid[r+i][c+j]:
                            grid[r+i][c+j]['is_placed'] = True
                rowspan_attr = f" rowspan=\"{cell['rowspan']}\"" if cell['rowspan'] > 1 else ""
                colspan_attr = f" colspan=\"{cell['colspan']}\"" if cell['colspan'] > 1 else ""
                cell_text = html.escape(cell['text'])
                html_content += f"    <td{rowspan_attr}{colspan_attr}>{cell_text}</td>\n"
        html_content += "  </tr>\n"
    html_content += "</table>"
    print("✅ Advanced HTML conversion complete!")
    return html_content

# --- Module 4.5: Main Orchestrator for Stage 4 ---
def convert_tables_to_html(table_dir, layout_dir, final_dir, available_gpus):
    """
    Orchestrates the entire advanced table conversion process.
    """
    print("\n" + "="*80 + "\n--- STAGE 4: Starting Advanced Table to HTML Conversion ---\n" + "="*80)
    
    # 1. Initialize VLM model once
    vlm_model, vlm_processor = initialize_vlm()
    if not vlm_model:
        print("❌ CRITICAL: VLM model failed to initialize. Aborting Stage 4.")
        # As a fallback, copy original layout jsons to final dir
        if not os.path.exists(final_dir) or not os.listdir(final_dir):
            shutil.copytree(layout_dir, final_dir, dirs_exist_ok=True)
        return

    # 2. Prepare final result directory by copying layout files
    if not os.path.exists(final_dir): os.makedirs(final_dir)
    for file in os.listdir(layout_dir):
        shutil.copy(os.path.join(layout_dir, file), os.path.join(final_dir, file))

    # 3. Process each cropped table image
    table_images = sorted([f for f in os.listdir(table_dir) if f.lower().endswith('.jpg')])
    for table_filename in tqdm(table_images, desc="Converting Tables to HTML"):
        table_image_path = os.path.join(table_dir, table_filename)
        tqdm.write(f"\nProcessing table: {table_filename}")
        
        # Extract cell data (slicing + OCR)
        cell_data = extract_cell_data(table_image_path, vlm_model, vlm_processor)
        if not cell_data:
            tqdm.write(f"  -> Skipping HTML conversion for {table_filename} as no cells were found.")
            continue
            
        # Convert structured cell data to a final HTML string
        html_table = convert_data_to_html_with_spans(cell_data)
        if not html_table:
            tqdm.write(f"  -> HTML generation failed for {table_filename}.")
            continue
            
        # 4. Update the corresponding layout JSON with the new HTML table
        match = re.match(r'(.+)_page_(\d+)_table_(\d+)\.jpg', table_filename)
        if not match:
            tqdm.write(f"  -> Could not parse filename {table_filename} to update JSON.")
            continue
        
        base_name, page_num, table_idx_on_page = match.groups()
        table_idx_on_page = int(table_idx_on_page)
        
        json_path = os.path.join(final_dir, f"{base_name}_page_{page_num}.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r+', encoding='utf-8') as f:
                    layout_data = json.load(f)
                    # Find the correct table block and update its content
                    table_blocks = [b for b in layout_data.get("parsing_res_list", []) if b.get("block_label") == "table"]
                    if table_idx_on_page < len(table_blocks):
                        # Find the original block by its index from cropping stage
                        original_block_index = -1
                        table_counter = 0
                        for idx, block in enumerate(layout_data["parsing_res_list"]):
                            if block.get("block_label") == "table":
                                if table_counter == table_idx_on_page:
                                    original_block_index = idx
                                    break
                                table_counter += 1
                        
                        if original_block_index != -1:
                            layout_data["parsing_res_list"][original_block_index]["block_content"] = html_table
                            f.seek(0)
                            f.truncate()
                            json.dump(layout_data, f, ensure_ascii=False, indent=4)
                            tqdm.write(f"  -> Successfully updated {os.path.basename(json_path)} with new HTML table.")
                        else:
                            tqdm.write(f"  -> ERROR: Could not find matching table block index {table_idx_on_page} in {os.path.basename(json_path)}.")
                    else:
                        tqdm.write(f"  -> ERROR: Table index {table_idx_on_page} is out of bounds for {os.path.basename(json_path)}.")
            except Exception as e:
                tqdm.write(f"  -> ERROR updating JSON file {json_path}: {e}")
        else:
            tqdm.write(f"  -> WARNING: Corresponding JSON file not found at {json_path}.")

    # 5. Cleanup
    del vlm_model
    del vlm_processor
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"✅ STAGE 4 Complete: Processed {len(table_images)} tables.")


# ======================================================================================
# --- STAGE 4.5: AI-Powered Table Merging ---
# ======================================================================================
def get_row_column_count(row):
    """Helper function to count columns in a table row, accounting for colspan."""
    count = 0
    if row:
        for cell in row.find_all(['td', 'th']):
            try:
                count += int(cell.get('colspan', 1))
            except (ValueError, TypeError):
                count += 1
    return count

def intelligent_crude_merge(table_fragments):
    """Performs a smarter programmatic merge before sending to AI for revision."""
    print(" -> Performing intelligent crude merge...")
    base_soup = BeautifulSoup(table_fragments[0], 'lxml')
    base_tbody = base_soup.find('tbody')
    if not base_tbody:
        table_tag = base_soup.find('table')
        if not table_tag: table_tag = base_soup
        base_tbody = base_soup.new_tag('tbody')
        for row in table_tag.find_all('tr'):
            base_tbody.append(row.extract())
        table_tag.append(base_tbody)
    for next_html_fragment in table_fragments[1:]:
        next_soup = BeautifulSoup(next_html_fragment, 'lxml')
        next_rows = next_soup.find_all('tr')
        if not next_rows: continue
        last_row_base = base_tbody.find_all('tr')[-1] if base_tbody.find_all('tr') else None
        first_row_next = next_rows[0]
        if last_row_base and first_row_next:
            last_cell_base = last_row_base.find_all(['td', 'th'])[-1] if last_row_base.find_all(['td', 'th']) else None
            first_cell_next = first_row_next.find(['td', 'th'])
            if last_cell_base and first_cell_next and not first_cell_next.get_text(strip=True):
                if get_row_column_count(last_row_base) == get_row_column_count(first_row_next):
                    print(" -> Detected possible text continuation. Merging cells.")
                    for cell in first_row_next.find_all(['td', 'th'])[1:]:
                        last_cell_base.append(' ' + cell.get_text(strip=True))
                    next_rows.pop(0)

        for row in next_rows:
            base_tbody.append(row)
    return str(base_soup)

def ai_merge_consecutive_tables(final_results_dir):
    """
    Identifies sequences of adjacent tables and uses an AI process to merge them.
    """
    print("\n" + "="*80 + "\n--- STAGE 4.5: AI-Powered Intelligent Table Merging ---\n" + "="*80)
    all_pages_data = []
    json_files = sorted([f for f in os.listdir(final_results_dir) if f.endswith('.json')])
    def get_page_number(filename):
        match = re.search(r'_page_(\d+)', filename)
        return int(match.group(1)) if match else 0
    json_files.sort(key=get_page_number)

    for filename in json_files:
        with open(os.path.join(final_results_dir, filename), 'r', encoding='utf-8') as f:
            all_pages_data.append(json.load(f))

    all_blocks = [{'page_idx': p_idx, 'block_idx': b_idx, **b} for p_idx, p_data in enumerate(all_pages_data) for b_idx, b in enumerate(p_data.get('parsing_res_list', []))]
    i = 0
    while i < len(all_blocks):
        if all_blocks[i]['block_label'] == 'table':
            candidate_group = [all_blocks[i]]
            j = i + 1
            while j < len(all_blocks) and all_blocks[j]['block_label'] == 'table':
                candidate_group.append(all_blocks[j]); j += 1
            if len(candidate_group) > 1:
                print(f"Found a sequence of {len(candidate_group)} tables. Asking AI for merge decision.")
                table_fragments = [block['block_content'] for block in candidate_group]
                decision_prompt = f"Analyze the following HTML table fragments. Is the second a continuation of the first? Fragments: {json.dumps(table_fragments, ensure_ascii=False, indent=2)}. Respond ONLY with JSON: {{\"should_merge\": true/false}}."
                try:
                    client = OpenAI(api_key=Config.API_KEY, base_url=Config.API_BASE_URL)
                    completion = client.chat.completions.create(model=Config.AI_MODEL_NAME, messages=[{"role": "user", "content": decision_prompt}], temperature=0.0, response_format={"type": "json_object"})
                    decision = json.loads(completion.choices[0].message.content)

                    if decision.get("should_merge"):
                        print(" -> AI Decision: MERGE. Performing intelligent crude merge and sending for AI revision.")
                        crudely_merged_html = intelligent_crude_merge(table_fragments)
                        revision_prompt = f"""Your task is to REVISE and PERFECT this table. The following HTML was crudely merged and may have errors. Fix any structural issues, broken rows, or incorrect headers. **Crudely Merged Table to Revise:** ```html\n{crudely_merged_html}\n```... Please return ONLY a single valid JSON object with one key, "revised_html", containing the final, perfectly revised HTML table."""
                        revision_completion = client.chat.completions.create(model=Config.AI_MODEL_NAME, messages=[{"role": "user", "content": revision_prompt}], temperature=0.0, response_format={"type": "json_object"})
                        revised_content = json.loads(revision_completion.choices[0].message.content)

                        if "revised_html" in revised_content and revised_content["revised_html"]:
                            print(" -> AI revision successful. Updating document structure.")
                            final_html = revised_content["revised_html"]
                            first_block_info = candidate_group[0]
                            all_pages_data[first_block_info['page_idx']]['parsing_res_list'][first_block_info['block_idx']]['block_content'] = final_html
                            for block_to_deactivate in candidate_group[1:]:
                                all_pages_data[block_to_deactivate['page_idx']]['parsing_res_list'][block_to_deactivate['block_idx']]['block_label'] = 'merged_into_previous'
                                all_pages_data[block_to_deactivate['page_idx']]['parsing_res_list'][block_to_deactivate['block_idx']]['block_content'] = ''
                    else:
                        print(" -> AI Decision: DO NOT MERGE.")
                except Exception as e:
                    print(f" -> ERROR during AI table merge call: {e}")
                i = j
            else:
                i += 1
        else:
            i += 1
    print("\nSaving updated results after table merging...")
    for i, page_data in enumerate(tqdm(all_pages_data, desc="Saving merged files")):
        filepath = os.path.join(final_results_dir, json_files[i])
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(page_data, f, ensure_ascii=False, indent=4)
    print("✅ STAGE 4.5 Complete.")


# ======================================================================================
# --- STAGE 5: Post-processing and Aggregation ---
# ======================================================================================
def post_process_and_combine_results(results_dir, master_output_dir):
    """Cleans, combines results, and extracts a title TOC."""
    print("\n" + "="*80 + "\n--- STAGE 5: Post-processing and Aggregation ---\n" + "="*80)
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
            for i, block in enumerate(data.get("parsing_res_list", [])):
                if 'title' in block.get("block_label", ""):
                    titles_toc.append({"page_index": page_number, "block_index_on_page": i, "title_content": block['block_content']})
            all_pages_content.append({"input_path": data.get("input_path"), "parsing_res_list": data.get("parsing_res_list", [])})
        except Exception as e:
            tqdm.write(f"❗️ Error processing {filename}: {e}")
    final_combined_data = {"document_content": all_pages_content, "titles_toc": titles_toc}
    final_output_path = os.path.join(master_output_dir, "_combined_document.json")
    try:
        with open(final_output_path, 'w', encoding='utf-8') as f:
            json.dump(final_combined_data, f, ensure_ascii=False, indent=4)
        print(f"\n✅ STAGE 5 Complete. Aggregated results saved to: ➡️ {final_output_path}")
    except Exception as e:
        print(f"❌ ERROR in Stage 5: Could not save final file. Details: {e}")


# ======================================================================================
# --- Main Execution Orchestrator ---
# ======================================================================================
def main():
    """Main function to orchestrate the entire PDF table extraction pipeline."""
    pipeline_start_time = time.time()
    pdf_basename = os.path.splitext(os.path.basename(Config.PDF_PATH))[0]
    master_output_dir = os.path.join(os.getcwd(), f"output_{pdf_basename}")
    dirs = {
        "images": os.path.join(master_output_dir, '1_page_images'),
        "layout": os.path.join(master_output_dir, '2_layout_ocr_json'),
        "tables": os.path.join(master_output_dir, '3_cropped_tables'),
        "final": os.path.join(master_output_dir, '4_final_results_with_html')
    }
    print(f"🚀 Starting PDF Table Extraction Pipeline for: {Config.PDF_PATH}")
    for d in dirs.values(): os.makedirs(d, exist_ok=True)

    # --- Execute Pipeline Stages ---
    if not convert_pdf_to_images(Config.PDF_PATH, dirs["images"]): return
    if not analyze_document_layout(dirs["images"], dirs["layout"]): return
    num_tables = crop_tables_from_images(dirs["images"], dirs["layout"], dirs["tables"])
    if num_tables > 0:
        print("\n" + "="*80 + "\n--- Pre-Stage 4: Preparing for VLM Inference ---\n" + "="*80)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        available_gpus = get_available_gpus(Config.REQUIRED_FREE_MEM_GB)
        if not available_gpus:
            print("\n❌ CRITICAL ERROR: No free GPUs found. Cannot proceed."); return
        # This now calls the new, advanced table processing logic
        convert_tables_to_html(dirs["tables"], dirs["layout"], dirs["final"], available_gpus)
    else:
        print("No tables found. VLM processing will be skipped.")
        # If no tables, copy layout JSONs to final dir to ensure pipeline continuity
        if not os.path.exists(dirs["final"]) or not os.listdir(dirs["final"]):
            shutil.copytree(dirs["layout"], dirs["final"], dirs_exist_ok=True)

    ai_merge_consecutive_tables(dirs["final"])
    post_process_and_combine_results(dirs["final"], master_output_dir)
    pipeline_elapsed_time = time.time() - pipeline_start_time

    print("\n" + "#"*80)
    print("🎉 PIPELINE COMPLETE!")
    print(f"Total execution time: {pipeline_elapsed_time:.2f} seconds")
    print(f"Final combined results are in: ➡️ {os.path.join(master_output_dir, '_combined_document.json')}")
    print("#"*80)

if __name__ == "__main__":
    # 'spawn' is recommended for CUDA applications in multiprocessing
    set_start_method("spawn", force=True)
    main()
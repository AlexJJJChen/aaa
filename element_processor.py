# -*- coding: utf-8 -*-
# ======================================================================================
# OMNI-PARSER: ELEMENT PROCESSOR MODULE
#
# 负责裁剪视觉元素（阶段3）并识别其内容（阶段4）。
# Responsible for cropping visual elements (Stage 3) and recognizing their content (Stage 4).
# Logic is a 1:1 copy from the original omni_parser.py.
# ======================================================================================
import os
import json
import re
import time
import cv2
import html
from PIL import Image
from tqdm import tqdm

from utils import calculate_iou
from model_engine import (
    batch_recognize_with_openai_vision,
    batch_recognize_text_with_qwen,
    recognize_text_with_qwen_single,
    batch_recognize_tables_with_nanonets,
    recognize_table_with_nanonets_single,
    clean_vlm_html_response
)

# --- STAGE 3: Visual Element Cropping ---
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
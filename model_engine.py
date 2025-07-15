# -*- coding: utf-8 -*-
# ======================================================================================
# OMNI-PARSER: MODEL AND RECOGNITION ENGINE (STRICT 1:1 SPLIT)
#
# 负责按需加载模型、初始化客户端以及执行所有AI识别任务。
# 所有函数均直接从 omni_parser.py 拆分，未做任何逻辑修改。
# ======================================================================================
import os
import requests
import re
import cv2
import json
from openai import OpenAI
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入通用工具
from utils import encode_image_to_base64

# --- 模型初始化和辅助函数 ---
# 以下所有函数均直接从 omni_parser.py 复制

def initialize_local_models(config):
    """Initializes and returns all required local models based on selector."""
    models = {'qwen': None, 'nanonets': None}
    selector = config.ModelSelector
    
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
    # 检查是否有任何任务使用了GPT模型
    if any('gpt' in str(m).lower() for m in vars(selector).values() if isinstance(m, str)):
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
        # 严格遵循 omni_parser.py 的 URL 构造方式
        response = requests.post(
            f"{str(client.base_url).rstrip('/')}/chat/completions",
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
    """Uses a thread pool to process a batch of images concurrently."""
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

def clean_vlm_html_response(response_text):
    """Cleans the VLM's response to extract pure HTML."""
    match = re.search(r'```(?:html)?\s*(.*?)\s*```', response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response_text.strip().strip('"').strip("'")

def batch_recognize_text_with_qwen(image_batch, model, processor,prompt="直接提取图片中的所有文字内容。注意如果是空白的图片的话返回 ''。"):
    """Recognizes text from a BATCH of images using the Qwen-VL model."""
    if not image_batch: return []
    try:
        pil_images = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in image_batch]
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
    """Recognizes text from a SINGLE image (Qwen). Used as a reliable fallback for batch failures."""
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
    """Recognizes tables from a BATCH of images using Nanonets and returns HTML strings."""
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
    """Recognizes a table from a SINGLE image using Nanonets. Used for retry."""
    if image_np is None or image_np.size == 0:
        return "<table><tr><td>Error: Invalid image provided.</td></tr></table>"
    try:
        result = batch_recognize_tables_with_nanonets([image_np], model, processor, tokenizer)
        return result[0] if result and result[0] != "<BATCH_FAILURE>" else "<table><tr><td>Error: Nanonets single recognition failed.</td></tr></table>"
    except Exception as e:
        return f"<table><tr><td>Error: Nanonets single recognition failed. Details: {e}</td></tr></table>"
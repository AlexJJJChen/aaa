# -*- coding: utf-8 -*-
# ======================================================================================
# OMNI-PARSER: LAYOUT ANALYZER MODULE
#
# 使用PP-StructureV3对页面图像进行布局分析。
# Performs layout analysis on page images using PP-StructureV3.
# Logic is a 1:1 copy from the original omni_parser.py.
# ======================================================================================
import os
import time
import gc
from tqdm import tqdm

def run_step_2_layout_analysis(input_dir, output_dir, batch_size):
    """Performs layout analysis on images using PaddleX and saves results as JSON."""
    print("\n" + "="*80 + "\n--- STAGE 2: Layout Analysis ---\n" + "="*80)
    os.makedirs(output_dir, exist_ok=True)
    
    pipeline = None
    try:
        from paddlex import create_pipeline as pp_create_pipeline
        print("正在初始化 PP-StructureV3 pipeline...")
        pipeline = pp_create_pipeline(pipeline="layout_parsing", batch_size=batch_size)
        print("✅ Pipeline 初始化成功。")
        
        start_time = time.time()
        image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"发现 {len(image_files)} 张图片。开始处理...")

        for filename in tqdm(image_files, desc="分析布局"):
            input_path = os.path.join(input_dir, filename)
            try:
                # The predict method returns a generator
                output_generator = pipeline.predict(
                    input=input_path,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )
                
                base_filename = os.path.splitext(filename)[0]
                
                # Iterate through the generator (usually just one result)
                for i, result_object in enumerate(output_generator):
                    json_save_path = os.path.join(output_dir, f"{base_filename}.json")
                    result_object.save_to_json(save_path=json_save_path)
                    # Typically, we only need the first result for a single image
                    if i == 0:
                        break

            except Exception as e:
                tqdm.write(f" [ERROR] 处理 {filename} 时发生错误: {e}")

        elapsed_time = time.time() - start_time
        print(f"✅ STAGE 2 完成: 在 {elapsed_time:.2f} 秒内分析完所有图片。")
        print(f" ➡️ 输出保存在: {output_dir}")
        return True
    except ImportError:
        print("❌ 错误: 'paddlex' 未安装。请运行 'pip install paddlex'。")
        return False
    except Exception as e:
        print(f"❌ 错误: 初始化或运行 PaddleX pipeline 失败。细节: {e}")
        return False
    finally:
        if pipeline:
            del pipeline
        gc.collect()
        print("✅ PaddleX pipeline 资源已释放。")

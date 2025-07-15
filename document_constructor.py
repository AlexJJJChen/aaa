# -*- coding: utf-8 -*-
# ======================================================================================
# OMNI-PARSER: DOCUMENT CONSTRUCTOR MODULE
#
# 负责高级文档整合任务，包括表格合并、结果聚合和最终文档生成。
# Responsible for high-level document assembly tasks, including table merging,
# result aggregation, and final document generation.
# Logic is a 1:1 copy from the original omni_parser.py.
# ======================================================================================
import os
import json
import re
import html
from tqdm import tqdm
from openai import OpenAI


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
                decision = run_ai_analysis(client, model_name, decision_prompt)
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
                    revised_content = run_ai_analysis(client, model_name, revision_prompt)
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
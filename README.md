# OmniParser: Universal Document Analysis Pipeline V8.3
 
![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)
 
**OmniParser** is an expert-enhanced, multi-stage pipeline designed to deconstruct and understand complex documents. It transforms unstructured files such as PDFs, DOCX, PPTX, and TXT into structured, machine-readable formats like Markdown and JSON. By leveraging a hybrid approach of layout analysis, Visual Language Models (VLMs), and rule-based heuristics, OmniParser extracts text, tables, and figures with high fidelity.
 
This document provides a comprehensive guide to its architecture, setup, configuration, and usage.
 
---
 
## Table of Contents
 
-   [Key Features](#key-features)
-   [System Architecture & Workflow](#system-architecture--workflow)
-   [Installation & Setup](#installation--setup)
    -   [Prerequisites](#prerequisites)
    -   [Python Libraries](#python-libraries)
    -   [Local Models](#local-models)
-   [Configuration Guide](#configuration-guide)
    -   [File Paths & API](#file-paths--api)
    -   [The Model Selector](#the-model-selector)
    -   [Processing Parameters](#processing-parameters)
-   [How to Use](#how-to-use)
    -   [Running the Pipeline](#running-the-pipeline)
    -   [Input Formats](#input-formats)
    -   [Output Structure](#output-structure)
-   [Version 8.3 Highlights](#version-83-highlights)
-   [License](#license)
 
---
 
## Key Features
 
-   **Universal File Support**: Natively processes `.pdf`, `.docx`, `.pptx`, `.txt`, and `.zip` archives containing supported file types.
-   **Hybrid AI Approach**: Combines the power of multiple models (`gpt-4o`, `Qwen-VL`, `Nanonets-OCR`) for specialized tasks, ensuring optimal results for different content types.
-   **Advanced Layout Analysis**: Utilizes `PaddleX` for precise identification of text blocks, titles, images, and tables.
-   **Intelligent Table Recognition**:
    -   Differentiates between bordered and borderless tables, applying the best model for each.
    -   Reconstructs complex tables, including merged cells, from visual structure.
    -   Intelligently merges tables that are split across multiple pages using a combination of heuristics and AI analysis.
-   **Semantic Hierarchy Generation**: Employs a large language model to analyze extracted titles and generate a correct hierarchical structure (e.g., Chapter 1, Section 1.1).
-   **Resource Optimized (V8.3)**: Features an on-demand model loading mechanism that significantly reduces memory (VRAM) footprint and accelerates startup time. Models are only loaded when required by the pipeline.
-   **Comprehensive Output**: Generates a clean, final Markdown document alongside detailed JSON files containing structured data, bounding boxes, and metadata for each page.
 
---
 
## System Architecture & Workflow
 
OmniParser operates through a sequential, multi-stage pipeline. The specific workflow adapts based on the input file type.
 
### Standard Workflow (PDF, DOCX)
 
This is the most comprehensive workflow, involving all stages of the pipeline.
 
```
Input File (.pdf, .docx)
     │
     ▼
[Stage 1: Convert to Image]
     │
     ▼
[Stage 2: Layout Analysis] -> Identifies text, tables, images, titles per page.
     │
     ▼
[Stage 3: Crop & Deduplicate] -> Extracts table/image regions. Removes duplicate tables.
     │
     ▼
--- (On-Demand Model Loading) ---
     │
     ▼
[Stage 4: Recognition Engine]
     ├── (4a) Image Description (VLM)
     └── (4b) Table Recognition (Specialized Models)
     │
     ▼
[Stage 5: AI Table Merging] -> Merges tables split across pages.
     │
     ▼
[Stage 6: Aggregate Results] -> Combines all page data into a single document JSON.
     │
     ▼
[Stage 7: AI Hierarchy Analysis & Markdown Generation] -> Creates the final structured output.
     │
     ▼
Final Output (.md, .json)
```
 
### Specialized Workflows
 
-   **PPTX Workflow**: Converts each slide to an image and uses a VLM to generate a detailed description and summary for each slide, compiling the results into a final Markdown report.
-   **TXT Workflow**: Converts the text file into an image and uses a VLM to re-capture the content, automatically formatting it into a structured Markdown document with inferred titles.
 
---
 
## Installation & Setup
 
### Prerequisites
 
1.  **Python**: Ensure you have Python 3.8 or newer installed.
2.  **LibreOffice**: The pipeline requires a system-level installation of LibreOffice to convert `.docx` files to PDF.
    -   **Ubuntu/Debian**: `sudo apt-get install libreoffice`
    -   **CentOS/Fedora**: `sudo yum install libreoffice`
    -   **Windows/macOS**: Download and install from the [official website](https://www.libreoffice.org/download/download-libreoffice/). Ensure its program directory is added to your system's `PATH` environment variable.
 
### Python Libraries
 
Install the required Python packages using pip. It is highly recommended to use a virtual environment.
 
```bash
# Core dependencies for processing, AI, and image handling
pip install torch transformers openai requests Pillow opencv-python tqdm
 
# File conversion utilities
pip install pdf2image aspose-slides
 
# Layout analysis engine
pip install paddlex
 
# Optional but recommended for advanced table merging
pip install beautifulsoup4
```
 
### Local Models
 
For full offline functionality, download the required local models and place them in an accessible directory.
 
1.  **Qwen-VL**: Download from [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct).
2.  **Nanonets-OCR**: Download from [Hugging Face](https://huggingface.co/unum-cloud/nanonets-ocr-s).
 
Update the paths in the [Configuration](#configuration-guide) section to point to these models.
 
---
 
## Configuration Guide
 
All pipeline settings are managed within the `Config` class in `omni_parser.py`.
 
### File Paths & API
 
These are the primary settings you must configure.
 
```python
class Config:
    # --- V8核心变更: 输入文件路径 (V8 Core Change: Input File Path) ---
    # Set the absolute path to your input file (.pdf, .docx, .txt, .pptx, .zip)
    INPUT_PATH = "/path/to/your/document.pdf"
 
    # --- 本地模型路径 (Local Model Paths) ---
    # Update these paths to where you have stored the models
    VLM_MODEL_CHECKPOINT = "/path/to/your/Qwen2.5-VL-7B-Instruct"
    NANONETS_MODEL_CHECKPOINT = "/path/to/your/Nanonets-OCR-s"
 
    # --- OpenAI & 兼容API配置 (OpenAI & Compatible API Config) ---
    # Enter your API key and base URL (if using a proxy or compatible API)
    API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxx"
    API_BASE_URL = "[https://api.openai.com/v1](https://api.openai.com/v1)"
```
 
### The Model Selector
 
This powerful sub-class lets you assign the best model for each specific task. This is the core of the pipeline's flexibility.
 
-   **Available Options**: `'local_qwen'`, `'local_nanonets'`, `'gpt-4o'`, `'gpt-4.1-mini-2025-04-14'`
 
```python
    class ModelSelector:
        # Task 1: Image Description (Charts, photos, slides)
        IMAGE_DESCRIPTION = 'gpt-4o'
 
        # Task 2: Borderless Table Recognition
        BORDERLESS_TABLE_RECOGNITION = 'local_nanonets'
 
        # Task 3: Bordered Table Cell Recognition
        BORDERED_TABLE_CELL_RECOGNITION = 'local_qwen'
 
        # Task 4: Cross-Page Table Merging (Requires strong logical reasoning)
        TABLE_MERGING = 'gpt-4.1-mini-2025-04-14'
 
        # Task 5: Document Title Hierarchy Analysis
        TITLE_HIERARCHY = 'gpt-4.1-mini-2025-04-14'
```
 
### Processing Parameters
 
Fine-tune performance and resource usage.
 
```python
    # --- 处理参数 (Processing Parameters) ---
    PDF_TO_IMAGE_DPI = 200      # Higher DPI for better quality, larger files
    API_REQUEST_TIMEOUT = 120   # Timeout in seconds for API calls
    GPT4O_BATCH_SIZE = 10       # Concurrent requests to OpenAI API
    VLM_BATCH_SIZE = 16         # Batch size for local models (adjust based on VRAM)
    PADDLE_BATCH_SIZE = 16      # Batch size for PaddleX layout analysis
```
 
---
 
## How to Use
 
### Running the Pipeline
 
1.  **Configure**: Open `omni_parser.py` and set the `INPUT_PATH` and other parameters in the `Config` class as described above.
2.  **Execute**: Run the script from your terminal.
 
    ```bash
    python omni_parser.py
    ```
 
The script will automatically create an output directory and begin processing. Progress will be displayed in the console using `tqdm` progress bars for each stage.
 
### Input Formats
 
You can point `INPUT_PATH` to one of the following:
* A single file (`.pdf`, `.docx`, `.pptx`, `.txt`)
* A `.zip` file containing any combination of the supported file types. The script will extract and process each file sequentially.
 
### Output Structure
 
The script generates a master output directory named `output_<your_input_filename>` next to the input file. Inside, you will find:
 
```
output_<your_input_filename>/
├── 1_page_images/               # Images extracted from each page of the source document.
├── 2_layout_jsons/              # JSON files (one per page) with layout analysis results.
├── 3_cropped_tables/            # Image snippets of every table detected.
├── 3_cropped_images/            # Image snippets of every figure/photo detected.
├── temp_cells_for_batching/     # (Temporary) Cropped table cells for OCR.
├── _combined_document.json      # Aggregated JSON of all recognized content.
├── _document_with_hierarchy.json# The final JSON with title hierarchy information.
└── _final_document.md           # The final, structured Markdown output.
```
 
---
 
## Version 8.3 Highlights
 
This version introduces significant architectural improvements based on expert review and user feedback, focusing on resource management and performance.
 
-   **On-Demand Model Loading**: Models (`Qwen`, `Nanonets`, `OpenAI Client`) are no longer pre-loaded at startup. They are initialized "just-in-time" right before they are needed (Stage 4), dramatically reducing the initial memory footprint.
-   **Enhanced Performance**: By delaying the loading of large local models, preceding stages like layout analysis (Stage 2) can utilize full system resources (especially VRAM), restoring and exceeding previous performance benchmarks.
-   **Faster Startup & Lower Overhead**: The script starts much faster and uses significantly less memory, especially for workflows that do not require VLM recognition.
 
---
 
## License
 
This project is licensed under the MIT License. See the `LICENSE` file for details.

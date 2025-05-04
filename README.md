# VLM Finetune - Chart Similarity Evaluation

This project provides tools for fine-tuning and evaluating vision language models (VLMs) on chart similarity tasks. It focuses on training models to compare chart visualizations and determine their semantic similarity based on underlying data rather than visual appearance.

## Overview

The project uses the MiniCPM-V-2_6 model as a base and supports LoRA fine-tuning to improve performance on chart similarity evaluation tasks. The system analyzes pairs of chart images and outputs a similarity score along with a summary of differences.

## Features

- **Fine-tuning dataset preparation**: Tools to prepare datasets for fine-tuning VLMs
- **Model evaluation**: Evaluate model performance on chart similarity tasks
- **Single pair testing**: Test similarity between any two chart images
- **LoRA support**: Fine-tune using LoRA adapters for efficient training

## Installation

1. Clone the repository
2. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Structure

The dataset consists of pairs of chart images organized in directories:
- `finetune_dataset/`: Contains training data for fine-tuning
- `test_dataset/`: Contains test data for evaluation

Each visualization pair is stored in a directory (e.g., `VIS_1796/`) containing:
- `visualization.png`: The first chart image
- `ref_visualization.png`: The reference chart image for comparison

## Usage

### Dataset Preparation

Generate a fine-tuning dataset file from template:

```bash
python ds_cli.py generate-finetune-datafile \
  --dataset-dir ./finetune_dataset \
  --template-filename template.json \
  --dataset-filename dataset.json \
  --prompt-file ./prompts/comparison_prompt_minicpm-v2.6.txt
```

### Model Evaluation

Evaluate model performance on a test dataset:

```bash
python inf_cli.py evaluate \
  --dataset-dir ./test_dataset \
  --prompt-file ./prompts/comparison_prompt_minicpm-v2.6.txt \
  --use-lora \
  --lora-path ./path/to/lora/adapter \
  --model-name openbmb/MiniCPM-V-2_6 \
  --dtype float16 \
  --threshold 0.9
```

### Single Pair Testing

Test similarity between two specific images:

```bash
python inf_cli.py single-test-4-two-images \
  --image1 ./path/to/first/image.png \
  --image2 ./path/to/second/image.png \
  --prompt-file ./prompts/comparison_prompt_minicpm-v2.6.txt \
  --use-lora \
  --lora-path ./path/to/lora/adapter \
  --model-name openbmb/MiniCPM-V-2_6 \
  --dtype float16
```

### Create Default Prompt

Generate a default prompt file for chart comparison:

```bash
python inf_cli.py generate-legacy-prompt-file \
  --output-file ./prompts/custom_prompt.txt
```

## Evaluation Criteria

The model evaluates chart similarity based on:

1. **Axes and Labels**: Similarity of axis ranges, units, and labels
2. **Data Points**: Differences in key data points or trends
3. **Underlying Data**: Subtle differences that affect the charts' shapes or conclusions

The model ignores:
- Color differences between charts
- Order of data points (if x-axis uses discrete categories)
- Rotation of axes (x and y axes might be swapped)

## Output Format

The model outputs a JSON object containing:
- `similarity_score`: A float between 0 and 1 (higher means more similar)
- `difference_summary`: A concise description of significant differences

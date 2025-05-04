#!/usr/bin/env python
import os
import re
import json
import torch
import typer
from typing import Optional
from PIL import Image
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

app = typer.Typer(pretty_exceptions_show_locals=False)

def load_prompt_from_file(file_path: str) -> str:
    """Load prompt from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error loading prompt file: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return ""

def load_test_data(file_path: str) -> list:
    """Load test data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading test data: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return []

def initialize_model(use_lora: bool = False, lora_path: Optional[str] = None, 
                     model_name: str = 'openbmb/MiniCPM-V-2_6', 
                     dtype: str = 'float16') -> tuple:
    """Initialize the model based on configuration."""
    torch.manual_seed(0)

    # Set torch dtype based on parameter
    if dtype == 'bfloat16':
        torch_dtype = torch.bfloat16
    elif dtype == 'float16':
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # Load base model
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch_dtype)

    # Apply LoRA if specified
    if use_lora and lora_path:
        print(f"Loading LoRA adapter from: {lora_path}")
        lora_model = PeftModel.from_pretrained(model, lora_path, device_map="auto")
        model = lora_model.eval().cuda()
    else:
        model = model.eval().cuda()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    return model, tokenizer

def process_images(model, tokenizer, prompt: str, 
                   image_00_path: str, image_01_path: str, 
                   retry_count: int = 3) -> tuple:
    """Process a pair of images and return similarity score."""
    if not os.path.exists(image_00_path) or not os.path.exists(image_01_path):
        print(f"Image files not found: {image_00_path} or {image_01_path}")
        return None, None, False

    retry = retry_count
    while retry > 0:
        try:
            image_00 = Image.open(image_00_path).convert('RGB')
            image_01 = Image.open(image_01_path).convert('RGB')

            # Prepare message for model
            msgs = [{'role': 'user', 'content': [image_00, image_01, f'{prompt}']}]

            # Get model response
            answer = model.chat(image=None, msgs=msgs, tokenizer=tokenizer)

            # Extract similarity score
            match = re.search(r'"similarity_score"\s*:\s*([0-9]*\.?[0-9]+)', answer)
            if match:
                score = float(match.group(1))
                print(f"Extracted similarity_score: {score}")
                return score, answer, True
            else:
                print("similarity_score not found.")
                retry -= 1

        except Exception as e:
            print(f"Error processing images: {str(e)}")
            import traceback
            print(traceback.format_exc())
            retry -= 1

    if retry == 0:
        print('Failed to get score after multiple attempts')

    return None, None, False

@app.command()
def evaluate(
    dataset_dir: str = typer.Option("./test_dataset", help="Path to dataset directory"),
    dataset_filename: str = typer.Option("dataset.json", help="dataset file name"),
    prompt_file: str = typer.Option("./prompts/comparison_prompt_minicpm-v2.6.txt", help="Path to question file"),
    use_lora: bool = typer.Option(False, help="Whether to use LoRA model"),
    lora_path: Optional[str] = typer.Option(None, help="Path to LoRA adapter"),
    model_name: str = typer.Option("openbmb/MiniCPM-V-2_6", help="Model name or path"),
    dtype: str = typer.Option("bfloat16", help="Model data type (bfloat16, float16, float32)"),
    threshold: float = typer.Option(0.9, help="Similarity score threshold for correctness")
):
    """
    Evaluate image similarity using MiniCPM-V model with optional LoRA fine-tuning.
    """
    # Load question from file
    question = load_prompt_from_file(prompt_file)
    if not question:
        print(f"Failed to load question from {question_file}")
        return

    # Load test data
    test_data = load_test_data(os.path.join(dataset_dir, dataset_filename))
    if not test_data:
        print(f"Failed to load test data from {os.path.join(dataset_dir, dataset_filename)}")
        return

    # Initialize model
    model, tokenizer = initialize_model(use_lora, lora_path, model_name, dtype)

    # Process test data
    correct_cnt = 0
    total_cnt = 0

    for idx, item in enumerate(test_data):
        print('--------------------------------------')
        i_id = item['ID']
        human_rst = item['HUMAN']
        print(f'{idx}:{i_id}:{human_rst}')

        # Construct image paths
        image_00_path = f'{dataset_dir}/VIS_{i_id}/visualization.png'
        image_01_path = f'{dataset_dir}/VIS_{i_id}/ref_visualization.png'
        
        # Process images
        score, answer, success = process_images(
            model, tokenizer, question, image_00_path, image_01_path
        )

        if success:
            # Evaluate correctness
            if human_rst == 'G' and score >= threshold:
                correct_cnt += 1
                print('correct')
            elif human_rst == 'B' and score < threshold:
                correct_cnt += 1
                print('correct')
            else:
                print('not correct')

            total_cnt += 1
            print(answer)
        if total_cnt > 0:
            print(f'{total_cnt}:{correct_cnt}:{correct_cnt*1.0/total_cnt}')

    # Print final results
    if total_cnt > 0:
        print("\n=== Final Results ===")
        print(f"Total evaluated: {total_cnt}")
        print(f"Correct predictions: {correct_cnt}")
        print(f"Accuracy: {correct_cnt*1.0/total_cnt:.4f}")
    else:
        print("No images were successfully evaluated")


@app.command()
def single_test_4_two_images(
    image1: str = typer.Option(..., help="Path to first image"),
    image2: str = typer.Option(..., help="Path to second image"),
    prompt_file: str = typer.Option("./comparison_prompt_legacy.txt", help="Path to prompt file"),
    use_lora: bool = typer.Option(False, help="Whether to use LoRA model"),
    lora_path: Optional[str] = typer.Option(None, help="Path to LoRA adapter"),
    model_name: str = typer.Option("openbmb/MiniCPM-V-2_6", help="Model name or path"),
    dtype: str = typer.Option("float16", help="Model data type (bfloat16, float16, float32)")
):
    """
    Test image similarity on a single pair of images.
    """
    # Load prompt from file
    prompt = load_prompt_from_file(prompt_file)
    if not prompt:
        print(f"Failed to load prompt from {prompt_file}")
        return

    # Initialize model
    model, tokenizer = initialize_model(use_lora, lora_path, model_name, dtype)

    # Process images
    score, answer, success = process_images(
        model, tokenizer, prompt, image1, image2
    )

    if success:
        print("\n=== Results ===")
        print(f"Similarity score: {score}")
        print(f"Full response:\n{answer}")
    else:
        print("Failed to process images")

@app.command()
def generate_legacy_prompt_file(
    output_file: str = typer.Option("./comparison_prompt_legacy.txt", help="Path to output question file")
):
    """
    Create a default question file with the standard similarity evaluation prompt.
    """
    question = """calculate the semantic similarity score between <image_00> <image_01>, the semantic similarity is based on the follow criteria:
1. Axes and Labels: Are the axes ranges, units, and labels similar between the two charts?
2. Data Points: Are the key data points significantly different?
3. **ignore** the color difference across these two chart images when comparing them.
4. **ignore** data points order difference if x-axis labels are discrete values.
5. **ignore** the x-axis and y-axis maybe rotated 90 degrees between the two charts.
6. focus on **the subtle difference of underlying data** to form these two chart images carefully.
return the following information:
1. the similarity_score between 0 and 1, the higher the score, the more similar these two chart images
2. the difference_summary to describe the significant difference between these two chart images

only output json format data
Do **NOT** output any other information
"""

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(question)
        print(f"Question file created at: {output_file}")
    except Exception as e:
        print(f"Error creating question file: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    app()

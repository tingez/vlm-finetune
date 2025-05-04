#!/usr/bin/env python
import os
import re
import json
import typer

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


@app.command()
def generate_finetune_datafile(
    dataset_dir: str = typer.Option("../finetune_dataset", help="Path to dataset directory"),
    template_filename: str = typer.Option("template.json", help="template file name"),
    dataset_filename: str = typer.Option("dataset.json", help="output dataset file name"),
    prompt_file: str = typer.Option("./comparison_prompt_legacy.txt", help="Path to prompt file"),
):
    """
    Generate finetune data file from template and prompt file.
    """
    # generate finetune data file from template and prompt file
    template_file = os.path.join(dataset_dir, template_filename)
    dataset_file = os.path.join(dataset_dir, dataset_filename)

    prompt = load_prompt_from_file(prompt_file)
    if not prompt:
        print(f"Failed to load prompt from {prompt_file}")
        return

    # load template
    template_fd = open(template_file, 'r', encoding='utf-8')
    template_json = json.load(template_fd)
    template_fd.close()
    for item in template_json:
        item['conversations'][0]['content'] = prompt
        item['image']['<image_00>'] = os.path.join(dataset_dir, item['image']['<image_00>'])
        item['image']['<image_01>'] = os.path.join(dataset_dir, item['image']['<image_01>'])
    
    
    # save template
    dataset_fd = open(dataset_file, 'w', encoding='utf-8')
    json.dump(template_json, dataset_fd, indent=4)
    dataset_fd.close()

if __name__ == "__main__":
    app()

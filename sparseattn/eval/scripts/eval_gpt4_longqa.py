import argparse
import json
import os
import sys
import re
from tqdm import tqdm
import glob

from typing import Optional, Any, Dict, List
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import functools

from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

import time

# Get the parent directory path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the parent directory to the Python path
sys.path.append(parent_dir)

# Import shared utilities
from gpt4_eval_utils import LLM, OpenAIModel, TgiVllmModel, format_chat, parse_output, parse_json, check_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--data_file", type=str, default=None, 
                        help="Path to input data file (JSON format)")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to output metrics file (defaults to data_file + '_metrics.json')")
    parser.add_argument("--results_file", type=str, default=None,
                        help="Path to detailed results file (defaults to data_file + '_results.jsonl')")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory to process all JSON files (alternative to --data_file)")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_length", type=int, default=32000)
    parser.add_argument("--generation_max_length", type=int, default=2048)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--use_batch_api", action="store_true", help="Use OpenAI batch API for cheaper processing")
    parser.add_argument("--base_url", type=str, default=None, help="Base URL for TGI/vLLM endpoint")
    
    args = parser.parse_args()
    
    # Determine data files to process
    data_files = []
    if args.data_file:
        data_files = [args.data_file]
    elif args.input_dir:
        data_files = glob.glob(os.path.join(args.input_dir, "*.json"))
        data_files = [f for f in data_files if not f.endswith(('_metrics.json', '_results.jsonl', '-gpt4eval_o.json'))]
    else:
        # Default: look for JSON files in current directory
        data_files = glob.glob("*.json")
        data_files = [f for f in data_files if not f.endswith(('_metrics.json', '_results.jsonl', '-gpt4eval_o.json'))]
        
    if not data_files:
        logger.error("No data files found. Please specify --data_file or --input_dir, or run in a directory with JSON files.")
        return
    
    logger.info(f"Processing {len(data_files)} files: {data_files}")
    
    # Initialize model
    if args.base_url:
        model = TgiVllmModel(
            model_name=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_length=args.max_length,
            generation_max_length=args.generation_max_length,
            do_sample=args.do_sample,
            seed=args.seed,
            base_url=args.base_url,
        )
    else:
        model = OpenAIModel(
            model_name=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_length=args.max_length,
            generation_max_length=args.generation_max_length,
            do_sample=args.do_sample,
            seed=args.seed,
        )
    
    # Process each data file
    for data_file in data_files:
        logger.info(f"Processing {data_file}")
        
        # Set output files based on data file if not provided
        if args.output_file:
            output_file = args.output_file
        else:
            output_file = data_file.replace('.json', '-gpt4eval_o.json')
            
        if args.results_file:
            results_file = args.results_file
        else:
            results_file = data_file.replace('.json', '_results.jsonl')
        
        # Skip if output already exists
        if os.path.exists(output_file):
            logger.info(f"Output file {output_file} already exists, skipping...")
            continue
        
        # Load data
        try:
            with open(data_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading {data_file}: {e}")
            continue
        
        # Handle different data formats
        if isinstance(data, dict) and 'data' in data:
            examples = data['data']
        elif isinstance(data, list):
            examples = data
        else:
            logger.error(f"Unsupported data format in {data_file}")
            continue
        
        # Process examples
        results = []
        for example in tqdm(examples, desc=f"Processing {os.path.basename(data_file)}"):
            try:
                # Prepare prompt - handle different data formats
                prompt = example.get('prompt', example.get('question', example.get('input_prompt', '')))
                
                if not prompt:
                    logger.warning(f"No prompt found in example {len(results)}")
                    continue
                
                # Generate response
                output = model.generate(prompt=prompt)
                
                if output:
                    result = {
                        'example_id': example.get('id', len(results)),
                        'prompt': prompt,
                        'generated_output': output['output'],
                        'input_len': output['input_len'],
                        'output_len': output['output_len'],
                        'reference': example.get('answer', example.get('reference', example.get('reference_output', ''))),
                    }
                    
                    # Parse output if needed
                    parsed_output = parse_output(output['output'])
                    result['parsed_output'] = parsed_output
                    
                else:
                    result = {
                        'example_id': example.get('id', len(results)),
                        'prompt': prompt,
                        'generated_output': None,
                        'error': 'Generation failed',
                        'reference': example.get('answer', example.get('reference', example.get('reference_output', ''))),
                    }
                    
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing example {len(results)}: {e}")
                results.append({
                    'example_id': len(results),
                    'error': str(e),
                })
        
        # Save results
        with open(results_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        
        # Compute and save metrics
        metrics = check_metrics(model, results_file, output_file)
        
        logger.info(f"Completed {data_file}. Results saved to {results_file}, metrics saved to {output_file}")


if __name__ == "__main__":
    main()
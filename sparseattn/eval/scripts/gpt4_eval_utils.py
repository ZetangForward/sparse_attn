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


def format_chat(
    message: str,
    system_message: Optional[str]=None,
) -> List[Dict[str, str]]:
    """
    Format the message into a list of dictionaries with role and content keys.
    This is useful for the chat-based models without tokenizer that does this.
    """
    if system_message is not None:
        chat = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": message},
        ]
    else:
        chat = [{"role": "user", "content": message}]
    return chat


class LLM:
    """
    Base class for generative models.
    """
    def __init__(
        self,
        model_name: str,
        temperature: float=0.9,
        top_p: float=0.9,
        max_length: int=32768,
        generation_max_length: int=2048,
        generation_min_length: int=0,
        do_sample: bool=True,
        stop_newline: bool=False,
        use_chat_template: bool=False,
        system_message: Optional[str]="You are a helpful assistant.",
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_length = max_length
        self.generation_max_length = generation_max_length
        self.generation_min_length = generation_min_length
        self.do_sample = do_sample
        self.use_chat_template = use_chat_template
        self.system_message = system_message
        self.stops = None
        if stop_newline:
            self.stops = ["\n", "\n\n"]

    """
    Prepare the data for input to the llm

    test_item: dict[str, any]
        the test item to be used for the generation, this dictionary is from the data preprocessing step and are used for further formatting to specific models, such as tokenization and/or chat formatting
    data: dict[str, any]
        the data dictionary that contains the template for the user message and system

    Returns the prepared input (type is model-specific)
    """
    def prepare_inputs(self, test_item: Dict[str, Any], data: Dict[str, Any]) -> Any:
        raise NotImplementedError("prepare_inputs not implemented for LLM")

    """
    Generate the output from the model

    The inputs have been prepared, the prompt is only the user message as a string that needs to be pre-processed.
    kwargs contains any additional parameters.
    This function should be implemented by the children class.

    The output should be a dictionary with the following:
     - "output" (str): the generated output
     - "input_len" (int): the length of the input tokens
     - "output_len" (int): the length of the output tokens
     - "input_text" (str or List[Dict[str, str]]): the input text or the chat format
    There may be additional keys depending on the model.
    This function may also return None in case of errors (e.g., denied by the API provider).

    """
    def generate(self, inputs: Optional[Any]=None, prompt: Optional[str]=None, **kwargs) -> Optional[Dict[str, Any]]:
        raise NotImplementedError("generate not implemented for LLM")

    """
    Generate the output from the model for a list of inputs or prompts.
    This is similar to to the generate function but everything is in a list.

    The children classes may override this function for optimization.
    """
    def generate_batch(self, inputs: Optional[List[Any]]=None, prompt: Optional[List[str]]=None, **kwargs) -> List[Optional[Dict[str, Any]]]:
        outputs = []
        if inputs is None:
            for p in tqdm(prompt):
                outputs.append(self.generate(prompt=p, **kwargs))
        else:
            for i in tqdm(inputs):
                outputs.append(self.generate(inputs=i, **kwargs))
        return outputs


class OpenAIModel(LLM):
    def __init__(
        self,
        model_name,
        temperature=0.9,
        top_p=0.9,
        max_length=32768,
        generation_max_length=2048,
        generation_min_length=0,
        do_sample=True,
        stop_newline=False,
        use_chat_template=True,
        system_message=None,
        seed=42,
        **kwargs,
    ):
        super().__init__(
            model_name,
            temperature=temperature,
            top_p=top_p,
            max_length=max_length,
            generation_max_length=generation_max_length,
            generation_min_length=generation_min_length,
            do_sample=do_sample,
            stop_newline=stop_newline,
            use_chat_template=use_chat_template,
            system_message=system_message,
        )
        import openai
        import tiktoken
        if "azure" in model_name:
            # env var: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and OPENAI_API_VERSION
            self.model = openai.AzureOpenAI()
            model_name = model_name[model_name.index("/")+1:]
        else:
            # make sure to set the OPENAI_API_KEY environment variable
            self.model = openai.OpenAI()
        self.model_name = model_name
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.seed = seed
        self.API_MAX_LENGTH = 128000 # this is defined by the OPENAI API


    def prepare_inputs(self, test_item, data):
        buffer = 100
        # we don't include system message to stay consistent with other models, which defaults to None
        prompt = format_chat(data["user_template"].format(**test_item), system_message=self.system_message)
        inputs = "\n".join([f"Role: {x['role']}\nContent: {x['content']}" for x in prompt])
        tokens = self.tokenizer.encode(inputs)
        input_len = len(tokens)

        if self.max_length > self.API_MAX_LENGTH:
            logger.warning(f"max_length {self.max_length} is greater than {self.API_MAX_LENGTH}, setting to {self.API_MAX_LENGTH}")
            self.max_length = self.API_MAX_LENGTH

        if input_len > self.max_length - self.generation_max_length - buffer:
            truncate_length = input_len - (self.max_length - self.generation_max_length - buffer)
            new_context = self.tokenizer.decode(self.tokenizer.encode(test_item["context"])[:-truncate_length])
            test_item["context"] = new_context
            prompt = format_chat(data["user_template"].format(**test_item), system_message=self.system_message)
        return prompt


    def generate(self, inputs=None, prompt=None, **kwargs):
        if inputs is None:
            # for system_message, set the self.system_message attribute
            inputs = format_chat(prompt, system_message=self.system_message)

        # kwargs can be used to pass additional parameters to the model: max_tokens, stop, etc.
        func = functools.partial(
            self.model.chat.completions.create,
            model=self.model_name,
            messages=inputs,
            max_tokens=self.generation_max_length,
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.seed,
            **kwargs,
        )
        try:
            response = func()
        except Exception as e:
            logger.error(f"Error in generation: {e}")
            return None

        output = response.choices[0].message.content
        input_len = response.usage.prompt_tokens
        output_len = response.usage.completion_tokens

        return {
            "output": output,
            "input_len": input_len,
            "output_len": output_len,
            "input_text": inputs,
        }

    def batch_api(self, inputs, batch_file, **kwargs):
        """
        Use batch API for cheaper and faster batch processing.
        https://platform.openai.com/docs/guides/batch
        """
        from openai import OpenAI
        client = OpenAI()

        # Upload the file to OpenAI for batch processing
        batch_input_file = client.files.create(
            file=open(batch_file, "rb"),
            purpose="batch"
        )

        # Submit the batch job
        batch_job = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": "batch job for evaluation"
            }
        )

        # You would need to poll for the batch job status and download results
        # This is a simplified version - in practice you'd need to implement polling
        logger.info(f"Batch job submitted: {batch_job.id}")
        return batch_job.id

    def generate_batch(self, inputs=None, prompt=None, **kwargs):
        """
        Process batch using openai batch api if available, otherwise fall back to sequential processing.
        """
        # For now, fall back to sequential processing
        # In the future, this could be optimized to use the batch API
        outputs = []
        if inputs is None:
            for p in tqdm(prompt):
                outputs.append(self.generate(prompt=p, **kwargs))
        else:
            for i in tqdm(inputs):
                outputs.append(self.generate(inputs=i, **kwargs))
        return outputs


class TgiVllmModel(OpenAIModel):
    def __init__(
        self, 
        model_name, 
        temperature=0.9, 
        top_p=0.9, 
        max_length=32768, 
        generation_max_length=2048, 
        generation_min_length=0, 
        do_sample=True, 
        stop_newline=False, 
        use_chat_template=True, 
        system_message=None,
        seed=42,
        **kwargs
    ):
        super().__init__(
            model_name,
            temperature=temperature,
            top_p=top_p,
            max_length=max_length,
            generation_max_length=generation_max_length,
            generation_min_length=generation_min_length,
            do_sample=do_sample,
            stop_newline=stop_newline,
            use_chat_template=use_chat_template,
            system_message=system_message,
            seed=seed,
            **kwargs
        )
        import openai
        # for TGI/vLLM, we use the openai client but with a different base URL
        base_url = kwargs.get("base_url", "http://localhost:8000/v1")
        self.model = openai.OpenAI(base_url=base_url, api_key="dummy")
        self.model_name = model_name
        # Use a simple tokenizer estimate for TGI/vLLM since we don't have tiktoken
        self.API_MAX_LENGTH = kwargs.get("api_max_length", 32768)

    def generate_batch(self, inputs=None, prompt=None, **kwargs):
        # TGI/vLLM might support different batch processing
        outputs = []
        if inputs is None:
            for p in tqdm(prompt):
                outputs.append(self.generate(prompt=p, **kwargs))
        else:
            for i in tqdm(inputs):
                outputs.append(self.generate(inputs=i, **kwargs))
        return outputs


def parse_output(output, prefix="Answer:"):
    def lstrip_string(s, sub):
        if s.startswith(sub):
            return s[len(sub):]
        else:
            return s

    output = output.strip()
    
    # if the output starts with a prefix, strip it
    output = lstrip_string(output, prefix).strip()
    output = lstrip_string(output, "Answer").strip()
    output = lstrip_string(output, ":").strip()
    output = lstrip_string(output, "answer:").strip()
    output = lstrip_string(output, "answer").strip()
    output = lstrip_string(output, ":").strip()
    output = lstrip_string(output, "The answer is").strip()
    output = lstrip_string(output, "the answer is").strip()
    output = lstrip_string(output, ":").strip()

    return output


def parse_json(text):
    """Extract JSON from text that might contain additional content."""
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Try to find JSON content between braces
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and start_idx <= end_idx:
        json_str = text[start_idx:end_idx+1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # If that fails, try parsing the entire text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse JSON from: {text}")
        return None


def check_metrics(model, results_file, output_file):
    """Process evaluation results and compute metrics."""
    try:
        with open(results_file, 'r') as f:
            results = [json.loads(line) for line in f]
    except FileNotFoundError:
        logger.error(f"Results file not found: {results_file}")
        return None
    
    metrics = {}
    total_examples = len(results)
    
    if total_examples == 0:
        logger.warning("No results found in file")
        return metrics
    
    # Extract and compute basic metrics
    successful_generations = [r for r in results if r.get('output') is not None]
    success_rate = len(successful_generations) / total_examples
    
    # Compute average lengths if available
    if successful_generations:
        avg_input_len = sum(r.get('input_len', 0) for r in successful_generations) / len(successful_generations)
        avg_output_len = sum(r.get('output_len', 0) for r in successful_generations) / len(successful_generations)
        
        metrics.update({
            'success_rate': success_rate,
            'total_examples': total_examples,
            'successful_generations': len(successful_generations),
            'avg_input_length': avg_input_len,
            'avg_output_length': avg_output_len,
        })
    
    # Save metrics
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Computed metrics: {metrics}")
    return metrics 
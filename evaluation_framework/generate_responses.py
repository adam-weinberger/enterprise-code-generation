#!/usr/bin/env python

# Imports
import os
import sys
import json
import termcolor
import requests
import pandas as pd
import os
import time
from tqdm import tqdm

# Shared config from across project
from config import config_dict, logger

# Global config, can be overridden via function arguments to 
# evaluation_framework function.
PROMPTS_PATH = config_dict['prompt_directory']
GENERATED_RESPONSES_DIRECTORY = config_dict['generated_responses_directory']
MODEL = config_dict['inference_model']
TEMPERATURE = config_dict['inference_temperature']
TOP_P = config_dict['inference_top_p']
TIMES = config_dict['inference_times']
MAX_TOKENS = config_dict['inference_max_tokens']
INFERENCE_ENDPOINT = config_dict['inference_endpoint']

metadata = {
    'MODEL': MODEL,
    'TEMPERATURE': TEMPERATURE,
    'TOP_P': TOP_P,
    'TIMES': TIMES,
    'MAX_TOKENS': MAX_TOKENS
}


# Function definitions
def run_completion_call(src_txt,
                        model,
                        max_tokens,
                        top_p,
                        temperature):
    """ Execute a single call to refact /v1/completions API

    Keyword arguments:
    src_txt -- prompt to be fed to refact hosted model. The Completion
    API assumes that prompt is a completion-oriented prompt, typically
    of the form <PRE> ... text ... <SUF> .. text .. <MID>. It is the 
    model's task to "fill in the missing text between <PRE> and <SUF>.
    """
    res = requests.post(f"{INFERENCE_ENDPOINT}/v1/completions", json={
        "model": model,
        "max_tokens": max_tokens,
        "stream": False,
        "echo": True,
        "top_p": top_p,
        "temperature": temperature,
        "prompt": src_txt,
        "stop": ["\n\n\n"],
    })
    res.raise_for_status()
    j = res.json()
    logger.info(j)
    return j["choices"][0]["text"]


def generate_responses(output_directory = GENERATED_RESPONSES_DIRECTORY,
                       prompts_path = PROMPTS_PATH,
                       model = MODEL,
                       temperature = TEMPERATURE,
                       top_p = TOP_P,
                       times = TIMES,
                       max_tokens = MAX_TOKENS):
    """ Given an output directory, generate responses via inference
    and writes responses to the output directory.

    Keyword arguments:
    output_directory -- directory to write responses, default specified
                        in config.py
    prompts_path -- directory to find prompt files. default specified
                    in config.py.
    prompt_file -- alternative, specify a specific prompt file. If
                   prompt file is specified, the batch directory
                   approach is overriden. No default.
    model -- specify model to be run format examples: 'codellama' or
             'codellama/7b/lora-20231107-201630' in the case of a fine
             tuned model. Default specified in config.py.
    temperature -- temperature for chosen model. Default specified in 
                   config.py
    top_p -- Top P cutoff for token selection. Default specified in 
             config.py
    times -- Number of inference calls for the prompt, supporting
             sampling of outputs. Default specified in config.py
    max_tokens - Maximum number of tokens to be generated. Default
                 specified in config.py.
    """
    # ToDo: Add in ability to specify a specific prompts file and output directory for testing
    logger.info("Generate responses")

    # Process all prompt files in prompts/ directory
    for file in os.listdir(prompts_path):
        if file.endswith('.csv') and file != 'skipped_files.csv':
            
            # Extract base filename without extension for later export
            base_filename = os.path.splitext(file)[0]

            logger.info(f"Generating responses for {file}")
            file_path = os.path.join(prompts_path, file)
            # Prompt dataframe. Columns are expected to be labeled
            df = pd.read_csv(file_path)

            # Loop over prompts dataframe, run inference to complete infilling
            # After each file is complete, write new .csv with inference output
            # as new column. Additionally write corresponding metadata file.
            completion_output = []
            for i in tqdm(range(0,len(df))):
             
                # prompt from input dataframe
                prompt = df.iloc[i]["prompt"]
                # print_prompt = termcolor.colored(prompt, "yellow") 

                #logger.info("\n", "Prompt:","\n",print_prompt)

                # run inference with prompt
                t = run_completion_call(prompt,
                                        model,
                                        max_tokens,
                                        top_p,
                                        temperature)

                # add response to list
                completion_output.append(t)

                # Print response together with prompt and label
            
                # print_response = termcolor.colored(t, "green")

                # logger.info("\n", "Inference output:", "\n", print_response)
                # logger.info("\n", "Label:","\n", df.iloc[i]["label"])

            # Add generated output back to original DataFrame
            df['completion'] = completion_output

            try:
                # Write inference output to .csv
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                model_filename = model.replace('/', '-')  # Replace forward slashes in model name
                inference_output_filename = os.path.join(output_directory, f"{timestamp}_{base_filename}_{model_filename}_output.csv")
            
                # Check if the directory exists, create if not
                if not os.path.exists(output_directory):
                    os.makedirs(output_directory)
            
                df.to_csv(inference_output_filename, index=False)
            
            except Exception as e:
                print(f"An error occurred while writing the CSV file: {e}")
            
            try:
                # Write metadata to a JSON file
                metadata_output_filename = os.path.join(output_directory, f"{timestamp}_{base_filename}_{model_filename}_metadata.json")
                
                with open(metadata_output_filename, 'w') as meta_file:
                    json.dump(metadata, meta_file, indent=4)
            
            except Exception as e:
                print(f"An error occurred while writing the JSON file: {e}")


if __name__ == "__main__":
    #ToDo update to allow override of prompts file and output directory for testing
    if len(sys.argv) >1:
        output_directory = sys.argv[1]

    # test function call to run stand-alone rather than as part of evaluation_framework.py
    generate_responses(output_directory = GENERATED_RESPONSES_DIRECTORY,
                       prompts_path = PROMPTS_PATH,
                       model = MODEL,
                       temperature = TEMPERATURE,
                       top_p = TOP_P,
                       times = TIMES,
                       max_tokens = MAX_TOKENS)

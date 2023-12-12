# Enterprise Code Generation

## Introduction

This repo implements an evaluation framework for testing a code generation model on your Java codebase. This framework assumes your model is hosted in [Refact](https://refact.ai/).

### Motivation

Although pretrained code-generating models are suitable for the average user, they are less so when it comes to writing code for enterprise. That is because pretrained models are only trained on data in the public domain and thus struggle to generate code that replicates the unique features of a private codebase (proprietary algorithms, coding standards, custom libraries, etc.). Consequently, many enterprises are looking to the in-house fine-tuned model as a better way to increase efficiency in software development. 

Solutions for fine-tuning code-generators are provided by projects like [Refact](https://refact.ai/), which takes a codebase and then fine-tunes a pretrained model using all the code contained. However, how do you evaluate the ability of a code generator to apply what it leared from fine-tuning? How do you compare different solutions created with the same private codebase? This project provides a java-oriented solution for achieving both these tasks. 

## How to use the tool
1) Follow the instructions in [requirements.txt](./requirements.txt) to create a conda environment with the appropriate libraries. Note there are a few manual steps
2) Edit the [configurations](./evaluation_framework/config.py) file in the evaluation_framework folder to have your desired settings.<br>
2.1. Importantly, define which modules in the pipeline you would like to run. <br>
2.2. If you are running generate_responses, you will need an instance of [Refact](https://refact.ai/) running with an endpoint for completions. You can set the address of the endpoint in the configs. The generate_responses code can be customized to work with a different endpoint<br>
2.3. The pass@k code needs to be customized to run the unit tests for each codebase.
3) From the parent directory, run the evaluation_framework/evaluation_framework.py script. Output for your run will apear in a timestamped subfolder inside the data folder


## Description of the pipeline
There are six steps for running the pipeline end-to-end, five of which are completed in this code and one of which must be done externally with [Refact](https://refact.ai/). The five steps (split train test, prompt maker, generate responses, evaluate responses, and analyze results) each have a separate `.py` file in the [evaluation_framework](./evaluation_framework) folder. All output data is stored and the data folder within the appropriate run_ts folder.

Configurations:

pipeline_steps_to_run: which of the five (not including Fine-tune Model) steps to run.

run_ts: if None creates a new run timestamp and puts all the data in that folder else continues with data from the specified run

### [Split Train Test](./evaluation_framework/split_train_test.py)

This module splits the files of a code base into train and test sets.

Configurations:

code_base_directory: relative file path to the code base from capstone-code-generation

train_ratio: The fraction of files in the training set

output_directory: The relative location of the data folder for outputs inside of capstone-code-generation

seed: random seed

split_strategy: methodology for making the train-test split

### (In [Refact](https://refact.ai/)) Fine-tune Model

Please see Refact documentation for up loading training and set, fine-tuning model, and hosting an endpoint.

### [Prompt Maker](./evaluation_framework/prompt_maker.py)

This module parses the test set into prompt-label pairs formatted for model ingestion. There are three masking strategies to create the prompts: token, line, and method.

num_label_tokens: number of consecutive tokens to mask

tokens_prompts_count: number of token prompts to create

num_label_lines: number of consecutive lines to mask

lines_prompts_count: number of line prompts to create

methods_prompts_count: number of method prompts to create

infill: whether or not to create infilling (prefix and suffix) or prefix only prompt

max_lines_above: number of lines of context above the masked code

max_lines_below: number of lines of context below the masked code

### [Generate Responses](./evaluation_framework/generate_responses.py)

This module sends the prompts to the Refact endpoint and records the responses (i.e. generated code)

inference_endpoint: ip address of endpoint

inference_model: name of model to use for code generation (must match name in Refact)

inference_temperature: amount of randomness or a creativity in generated response

inference_top_p: cumulative probability of tokens to consider at a time

inference_times: number of responses per prompt to generate (not currently implemented)

inference_max_tokens: maximum number of tokens to generate

### [Evaluate Responses](./evaluation_framework/evaluate_responses.py)

This module evaluates the performance of each generated response.

evaluation_metrics: list of evaluation metrics to calculate

evaluation_parallel_process_count: number of parallel processes to use for pass@k calculation


### [Analyze Results](./evaluation_framework/analyze_results.py)

This module calculates summary statistics for the performance.
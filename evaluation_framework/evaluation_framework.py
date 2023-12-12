#!/usr/bin/env python
import json
import os

from config import config_dict, logger
from datetime import datetime
from analyze_results import analyze_results
from evaluate_responses import evaluate_responses
from generate_responses import generate_responses
from prompt_maker import make_prompts
from split_train_test import split_train_test

if __name__ == "__main__":

    if not config_dict["run_ts"]:
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        run_ts = config_dict["run_ts"]

    run_folder_path = 'data/run_' + str(run_ts) + '/'
    config_dict["output_directory"] = run_folder_path
    config_dict["prompt_directory"] = config_dict["output_directory"] + "prompts/"
    config_dict["generated_responses_directory"] = config_dict["output_directory"] + "generated_responses/"
    config_dict["evaluation_directory"] = config_dict["output_directory"] + "evaluated_responses/"
    config_dict["analyzed_results_directory"] = config_dict["output_directory"] + "analyzed_results/"

    os.makedirs(os.path.dirname(run_folder_path), exist_ok=True)
    with open(f"{run_folder_path}config_dict.json", "w") as outfile:
        json.dump(config_dict, outfile, indent=4)

    logger.info(f'Beginning evaluation run {run_ts}')

    if 'split_train_test' in config_dict['pipeline_steps_to_run']:
        split_train_test(
            config_dict["code_base_directory"],
            config_dict["train_ratio"],
            config_dict["output_directory"],
            config_dict["seed"],
            config_dict["file_mapping_name"],
            config_dict["split_strategy"]
        )
        
  
    if 'make_prompts' in config_dict['pipeline_steps_to_run']:
        make_prompts(
            config_dict["output_directory"],
            config_dict["num_label_lines"],
            config_dict["lines_prompts_count"],
            config_dict["num_label_tokens"],
            config_dict["tokens_prompts_count"],
            config_dict["methods_prompts_count"],
            config_dict["infill"],
            config_dict["max_lines_above"],
            config_dict["max_lines_below"]
        )

    if 'generate_responses' in config_dict['pipeline_steps_to_run']:
        generate_responses(
            output_directory = config_dict["generated_responses_directory"],
            prompts_path =  config_dict["prompt_directory"],
            model = config_dict['inference_model'],
            temperature = config_dict["inference_temperature"],
            top_p = config_dict["inference_top_p"],
            times = config_dict["inference_times"],
            max_tokens = config_dict["inference_max_tokens"], 
            inference_endpoint=config_dict["inference_endpoint"]
        )
   
    if 'evaluate_responses' in config_dict['pipeline_steps_to_run']:
        evaluate_responses(
            config_dict["generated_responses_directory"],
            config_dict["evaluation_directory"], 
            config_dict["code_base_directory"],
            config_dict["evaluation_metrics"]
        )

    if 'analyze_results' in config_dict['pipeline_steps_to_run']:
        analyze_results(
            config_dict["evaluation_directory"], 
            config_dict["analyzed_results_directory"], config_dict["evaluation_metrics"]
        )
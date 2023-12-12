"""
Code should be run from the capstone-code-generation directory and the
codebase should be saved in a separate directory at the same level as
capstone-code-generation
"""
import os
import logging

config_dict = {
  #run
  "pipeline_steps_to_run": ["split_train_test", "make_prompts", "generate_responses", "evaluate_responses"],
  "run_ts": "run_20231212_130935",#None,
  #split_train_test
  "code_base_directory": "../kestra/",
  "train_ratio": 0.8,
  "output_directory": "data/",
  "seed": 0,
  "file_mapping_name": "file_mappings",
  "split_strategy": "random",
  #prompt_maker
  "num_label_tokens": 2,
  "tokens_prompts_count": 10,
  "num_label_lines": 2,
  "lines_prompts_count": 10,
  "methods_prompts_count": 10,
  "infill": True, 
  "max_lines_above": 50,
  "max_lines_below": 50,
  #generate_responses
  "inference_endpoint": "http://127.0.0.1:8008", 
  "inference_model": "codellama/7b",
  "inference_temperature": 0.2,
  "inference_top_p": 0.95,
  "inference_times": 1, 
  "inference_max_tokens": 256,
  #evaluate_responses
  "evaluation_metrics": ["CodeBLEU", "pass@k"],
  "evaluation_parallel_process_count": os.cpu_count() - 1
 }

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

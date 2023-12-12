import os
import numpy as np
import pandas as pd
import re
from scipy import stats

import javalang

from config import config_dict, logger

def margin_of_error(p_hat, n, confidence_level=0.95):
  """Calculates the margin of error for a proportion.

  Args:
    p_hat: The sample proportion.
    n: The sample size.
    confidence_level: The confidence level.

  Returns:
    The margin of error.
  """

  z = stats.norm.ppf(1 - (confidence_level / 2))
  se = np.sqrt(p_hat * (1 - p_hat) / n)
  return z * se

def count_tokens_in_label(label):
    label = str(label)
    label = label.replace('<EOT>', '')
    try:
        return len(list(javalang.tokenizer.tokenize(label)))
    except:
        return len(re.split(',| |_|-|=|\.|\+|\*', label))

def analyze_results_file(results_df, evaluation_metrics, base_modules=False):

    # All metrics are an a unless specified in evaluation_metrics
    successful_build_rate = np.nan
    successful_build_rate_moe = np.nan
    no_error_rate = np.nan
    no_error_rate_moe = np.nan
    pass_at_1_rate = np.nan
    pass_at_1_rate_moe = np.nan
    average_code_bleu_ = np.nan
    average_code_bleu_moe = np.nan

    if 'pass@k' in evaluation_metrics:
        successful_build_rate = (results_df['pass'] == 0).sum() / len(results_df)
        successful_build_rate_moe = margin_of_error(successful_build_rate, len(results_df))

        no_error_rate = ((results_df['pass'] == 0) & (results_df['error_count'] == 0)).sum() / len(results_df)
        no_error_rate_moe = margin_of_error(no_error_rate, len(results_df))

        pass_at_1_rate = ((results_df['pass'] == 0) & (results_df['failure_count'] == 0)).sum() / len(results_df)
        pass_at_1_rate_moe = margin_of_error(pass_at_1_rate, len(results_df))


    results_dict = {}
    columns_to_average = ['ngram_match_score', 'weighted_ngram_match_score', 'syntax_match_score', 'dataflow_match_score', 'label_token_count', 'completion_token_count']

    
    if base_modules:
        if 'CodeBLEU' in evaluation_metrics:
            average_code_bleu = 1
            average_code_bleu_moe = 0
        
        for column in columns_to_average:
            results_dict[f'average_{column}'] = np.nan

    else:
        if 'CodeBLEU' in evaluation_metrics:
            average_code_bleu = results_df['codebleu'].mean()
            average_code_bleu_moe = results_df['codebleu'].std() #  is this how you calculate the standard error for a bounded continuous variable?

        results_df['label_token_count'] = results_df['label'].apply(count_tokens_in_label)
        results_df['completion_token_count'] = results_df['completion'].apply(count_tokens_in_label)
        
        for column in columns_to_average:
            results_dict[f'average_{column}'] = results_df[column].mean()

    results_dict['successful_build_rate'] = successful_build_rate
    results_dict['successful_build_rate_moe'] = successful_build_rate_moe
    results_dict['no_error_rate'] = no_error_rate
    results_dict['no_error_rate_moe'] = no_error_rate_moe
    results_dict['pass_at_1_rate'] = pass_at_1_rate
    results_dict['pass_at_1_rate_moe'] = pass_at_1_rate_moe
    results_dict['average_code_bleu'] = average_code_bleu
    results_dict['average_code_bleu_moe'] = average_code_bleu_moe

    return results_dict


def analyze_results(evaluated_responses_directory, analyzed_results_directory, evaluation_metrics):
    logger.info("Analyzing results")

    results_dict = {}

    for file in os.listdir(evaluated_responses_directory):

        if file.startswith('base_module_build_results'):
            logger.info(f"Analyzing {file}")

            base_module_results_path = os.path.join(evaluated_responses_directory, file)
            modules_results_df = pd.read_json(base_module_results_path).transpose()
            modules_results_df['pass'] = modules_results_df['pass'].apply(lambda x: 0 if x else 1)
            results_dict['base_modules'] = analyze_results_file(modules_results_df, evaluation_metrics, True)
        
        if file.endswith('output.csv'): 
            logger.info(f"Analyzing {file}")
            
            results_file = os.path.join(evaluated_responses_directory, file)
            results_df = pd.read_csv(results_file)
            model = file.replace('.csv', '')
            results_dict[model] = analyze_results_file(results_df, evaluation_metrics)

            if 'lines' in model:
                results_dict[model]['prompt_type'] = 'lines' 
            elif 'tokens' in model:
                results_dict[model]['prompt_type'] = 'tokens' 
            elif 'methods' in model:
                results_dict[model]['prompt_type'] = 'methods' 

    results_df = pd.DataFrame(results_dict).transpose().sort_index()
    
    results_summary_path = os.path.join(analyzed_results_directory, 'results_summary.csv')
    os.makedirs(analyzed_results_directory, exist_ok=True)
    results_df.to_csv(results_summary_path)
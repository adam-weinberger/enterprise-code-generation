import json
from multiprocessing import Pool, Process
import os
from pathlib import Path
import re
import shutil
from typing import List, Tuple, Dict
import subprocess

from codebleu import calc_codebleu
from javalang.javalang import tokenizer
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import config_dict, logger


def evaluate_responses(generated_responses_directory: str, output_directory: str, code_base_directory: str , evaluation_metrics: List):
    """
    overwrite the real code with the generated code to create the set of generated code files
    Calculate the perplexity, compilation rate, CodeBLEU, and Pass@k metrics for the generated responses
    """
    logger.info("Evaluate responses")

    for file in os.listdir(generated_responses_directory):
        if file.endswith('.csv'):
            file_path = os.path.join(generated_responses_directory, file)

            logger.info(f"Evaluating {file_path}")
            responses_df = pd.read_csv(file_path)
            responses_df['completion'] = responses_df['completion'].str.replace('\n <EOT>', '')

            # probably not possible
            if "perplexity" in evaluation_metrics:
                _calculate_perplexity(responses_df)

            if "CodeBLEU" in evaluation_metrics:
                responses_df = _calculate_code_bleu(responses_df)

            if "pass@k" in evaluation_metrics:
                responses_df = _calculate_pass_at_k(responses_df, code_base_directory, output_directory)

            # file = file.replace("finetune_inference_output", "evaluated_responses")
            evaluated_responses_path = os.path.join(output_directory, file)
            logger.info(f'Writing evaluation results to {evaluated_responses_path}')
            os.makedirs(output_directory, exist_ok=True)
            responses_df.to_csv(evaluated_responses_path, index=False)


def _replace_text_in_file(file_path: str, search_text: str, replace_text: str) -> None:
    """
    Replace occurrences of a specified text in a file.

    :param str file_path: The path to the file to be processed.
    :param str search_text: The text to be replaced.
    :param str replace_text: The text to replace the occurrences of `search_text`.

    :return: None. The function modifies the specified file in-place.
    :rtype: None

    :raises Exception: If an error occurs during the file processing, an exception is caught,
                      and an error message is printed, indicating the file and the nature of the error.
    """
    try:
        with open(file_path, 'r') as file:
            file_content = file.read()
        
        # Use regular expressions to replace the search_text with replace_text
        # need to cast to strings because it's possible that the code is only numbers
        modified_content = file_content.replace(str(search_text), str(replace_text))

        with open(file_path, 'w') as file:
            file.write(modified_content)

    except Exception as e:
        print(f'Error while processing {file_path}: {e}')


def _calculate_perplexity(test_set):
    """
    Perplexity measures the confidence of the model in the choice of each generated token

    :param token_probabilities: a list of token conditional probabilities
    """
    logger.info("calculating perplexity")
    token_probabilities = np.array(test_set['token_probabilities'])
    n = len(token_probabilities)
    perplexity = np.power(np.prod(1 / token_probabilities[1:]), 1/n)

    return perplexity


def _javalang_codebleu_tokenizer_adapter(tokens):
    """
    Use the javalang parser which is better than the default parser in codebleu
    """
    tokens = list(tokenizer.tokenize(tokens))
    return [token.value for token in tokens]


def _calculate_code_bleu(test_set_df: pd.DataFrame):
    """
    Calculate the CodeBLEU score is a weighted combination of n-gram match (BLEU), weighted n-gram match (BLEU-weighted), AST match and data-flow match scores.
    The calculation is done by the codebleu package. Please see https://github.com/k4black/codebleu for details.
    """
    logger.info("Calculating CodeBLEU score")

    dict_list = []
    for index, row in test_set_df.iterrows():

        try:
            reference = row['label']
            prediction = row['completion']

            result = calc_codebleu(
                [reference],
                [prediction],
                lang="java",
                weights=(0.25, 0.25, 0.25, 0.25),
                tokenizer=_javalang_codebleu_tokenizer_adapter)
        except Exception as e:
            print(f'Error while calculating codebleu for {row["path_in_code_repo"]}: {e}')
            result = {
                 'codebleu': None,
                 'ngram_match_score': None,
                 'weighted_ngram_match_score': None,
                 'syntax_match_score': None,
                 'dataflow_match_score': None
                }
            
        dict_list.append(result)
    new_data = pd.DataFrame(dict_list)
    df_with_new_data = pd.concat([test_set_df, new_data], axis=1)
    return df_with_new_data


def _calculate_pass_at_k(responses_df: pd.DataFrame, code_base_directory: str, evaluation_directory: str):
    """
    find all of the modules
    for each module
        run maven clean install and capture the results
    for each file in the test set
        replace the code in the file
        maven install the file and capture the results
        compare the results to the module baseline
    """
    logger.info("Calculating Pass@k score")

    # we only want to run the tests in the same module as the generated code to save time
    # first find all the modules and get a baseline of which tests pass
    # this can take a very long time so it will only run if the codebase commit hash changed
    base_module_results_path = _get_base_module_results_path(code_base_directory, evaluation_directory)
    if os.path.exists(base_module_results_path):
        logger.debug("Base modules have already been built")
        with open(base_module_results_path, "r") as outfile: 
            module_results = json.load(outfile)
    else:
        module_results = _build_base_modules()
        with open(base_module_results_path, "w") as outfile: 
            json.dump(module_results, outfile, indent=4)
    

    # test each piece of generated code 
    logger.info("Building generated code and capturing results")
    
    # this is very slow to run this, so it is done in parallel
    responses_df_split = np.array_split(responses_df, config_dict['evaluation_parallel_process_count'])
    with Pool(config_dict['evaluation_parallel_process_count']) as p:
        generated_code_results = p.map(_build_generated_code, responses_df_split)

    # make a big dictionary of the results from each process, convert to a dataframe, and then append horizontally to the responses
    results_dict = {}
    for d in generated_code_results:
        for key, value in d.items():
            results_dict.setdefault(key, []).extend(value)

    generated_code_results = pd.DataFrame.from_dict(results_dict).sort_index()
    generated_code_results = pd.concat([responses_df.reset_index(drop=True), generated_code_results.reset_index(drop=True)], axis=1)

    return generated_code_results


def _get_base_module_results_path(code_base_directory, evaluation_directory) -> str:
    """
    The build results of the unaltered base modules are stored in a json dictionary.
    The name of the file has the git hash of the commi of the code base directory for versioning
    """
    current_working_directory = os.getcwd()
    os.chdir(code_base_directory)
    git_hash = subprocess.run([f'git rev-parse --short HEAD'], capture_output=True, text=True, shell=True)
    git_hash = git_hash.stdout.strip()
    os.chdir(current_working_directory)

    base_module_results_file = f"base_module_build_results_{git_hash}.json"
    base_module_results_path = os.path.join(evaluation_directory, base_module_results_file)

    return base_module_results_path


def _build_base_modules(code_base_directory: str = config_dict['code_base_directory']) -> Dict:

    logger.info("Finding and building base (unaltered) modules")
    modules = _find_modules(code_base_directory)
    module_results = {}
    build_from_scratch = True #  build the first module from scratch
    modules = modules
    for module in tqdm(modules):
        logger.info(module == 'salesforce')
        module_results[module] = _capture_maven_build_results(module, code_base_directory, build_from_scratch=build_from_scratch)
        # build_from_scratch = False

    return module_results


def _find_modules(directory: str) -> List[str]:
    modules = []
    with open(directory + "/pom.xml", 'r') as f:
        while (f.readline().strip() != "<modules>"): continue
        while ((line := f.readline().strip()) != "</modules>"):
            if (line.startswith("<module>") and line.endswith("</module>")):
                modules.append(line[8:-9])

    return modules


def _capture_maven_build_results(module: str, code_base_directory: str=config_dict['code_base_directory'], build_from_scratch: bool=False) -> Dict:
    
    logger.info(f"Building {module}")

    # run the maven command to build the module
    current_working_directory = os.getcwd()
    os.chdir(code_base_directory)
    build_from_scratch = " -am" if build_from_scratch else ""
    command = f'/opt/apache-maven-3.6.3/bin/mvn clean install -pl :{module}{build_from_scratch}'
    command_output = subprocess.run([command], capture_output=True, text=True, shell=True)
    logger.info(f'Result: {command_output.returncode}')

    result = {
    'pass': command_output.returncode == 0,
    'failure_count': 0,
    'failed_tests': [],
    'error_count': 0
    }

    # parse the standard out to count the number failures and record the names of the failed tests
    indicator = False
    for line in command_output.stdout.split('\n'):

        # record the number of failures and errors
        if indicator and "[ERROR] Tests run:" in line:
            failures_match = re.search(r"Failures: (\d+)", line)
            result['failure_count'] = int(failures_match.group(1))

            errors_match = re.search(r"Errors: (\d+)", line)
            result['error_count'] = int(errors_match.group(1))
            indicator = False
        
        # record the name of each failed test or error
        if indicator and  "[ERROR]   " in line:
            failed_test = line.split(' ')[3]
            result['failed_tests'].append(failed_test)

        # once we find the line "[ERROR] Failures: " or "[ERROR] Errors: " we can begin parsing the information related to the failed tests (aka indicator=True)
        if not indicator and (line == "[ERROR] Failures: " or line == "[ERROR] Errors: "):
            indicator = True

    os.chdir(current_working_directory)

    return result


def _build_generated_code(responses_df: pd.DataFrame, code_base_directory: str = config_dict['code_base_directory']) -> Dict:
    """
    Build and analyze generated code based on responses DataFrame.

    This function takes a DataFrame of responses, replaces code in the original file, builds the modified module,
    and performs analysis on the generated code. It returns a dictionary containing build results and additional
    failure information.

    :param pd.DataFrame responses_df: DataFrame containing responses with 'path_in_code_repo', 'label',
                                      and 'completion' columns.

    :return dict: A dictionary containing build results and additional failure information for each generated code entry.
                 The dictionary has the following structure:
                 {
                     '[index]': {
                         'additional_failure_count': int,
                         'additional_error_count': int,
                         'additional_failed_tests': List[str],
                         'additional_pass': bool,
                         # other build results...
                     },
                     # other entries...
                 }

    :raises AssertionError: If there are mismatches in failure counts or failed tests during the analysis,
                            an AssertionError is raised with details.
    """

    process_id = Process()._identity
    generated_code_results = {
        'pass': [],
        'failure_count': [],
        'failed_tests': [],
        'error_count': [],
        'module':[]
    }
    logger.info(f"Building generated code in process {process_id}")

    # sometimes the parallelization will send a dataframe with no rows
    if len(responses_df) > 0:

        # need a copy of the codebase for each process so they do not interfere with each other
        original_code_base_directory = code_base_directory
        primary_process_id = (process_id[0] % config_dict['evaluation_parallel_process_count']) + 1
        process_code_base_directory = f'../temp_code_base_directory_{primary_process_id}_{process_id[1]}/'

        if not os.path.exists(process_code_base_directory):
            logger.info(f'Copying code base to {process_code_base_directory}')
            shutil.copytree(original_code_base_directory, process_code_base_directory, dirs_exist_ok=True)
        else:
            logger.info(f'{process_code_base_directory} already exists')
        
        logger.info(f'Number examples to evaluate {len(responses_df)}')
        for index, row in tqdm(responses_df.iterrows()):

            # replace the original code in the original file in the code base
            path_in_code_repo = os.path.join(process_code_base_directory, row['path_in_code_repo'].replace(f'{code_base_directory}', ''))
            _replace_text_in_file(path_in_code_repo, row['label'], row['completion'])

            # build the module again
            module = re.search(fr"{process_code_base_directory}([^/]+)/", path_in_code_repo).group(1)
            generated_code_results['module'].append(module)
            # generated_code_results[str(index)] = _capture_maven_build_results(module, process_code_base_directory)
            # generated_code_results[str(index)]['module'] = module
            build_results_dict = _capture_maven_build_results(module, process_code_base_directory)
            generated_code_results['pass'].append(build_results_dict['pass'])
            generated_code_results['failure_count'].append(build_results_dict['failure_count'])
            generated_code_results['failed_tests'].append(build_results_dict['failed_tests'])
            generated_code_results['error_count'].append(build_results_dict['error_count'])

            # put the original code back in the module and continue with the next iteration of generated code
            _replace_text_in_file(path_in_code_repo, row['completion'], row['label'])

        # shutil.rmtree(process_code_base_directory)
        
        return generated_code_results


def _find_unit_test_files(directory: str) -> List[Tuple[str, str]]:
    """
    Search for Java unit test files within the specified directory and its subdirectories.

    :param directory: The directory to start the search from.
    :return: A list of paths to Java unit test files found.
    """
    unit_test_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.java') and 'test' in file.lower():
                unit_test_files.append((root, file))
    return unit_test_files


def _exact_match_unit_tests_with_app_files(app_files, unit_test_files: List[Tuple[str, str]]) -> Tuple[set, set, set]:
    """
    Match each file with the test file of the corresponding name. Exact matches only

    :param app_files: A pandas dataframe containing the path to each of the application files
    :param unit_test_files: A list of tuples of (path, filename)
    :return: A tuple of (files without tests, files with tests, tests without files)
    """

    # files which are not the unit tests
    non_test_files = set([Path(path).name for path in app_files['path'].tolist()])

    # need to make the unit test files into tuples (original file name, filename without the word test)
    # in order to properly match with non test files
    test_files = set()
    for path_file_tuple in unit_test_files:
        test_files.add((path_file_tuple[1], re.sub(r'test|Test|TEST', '', path_file_tuple[1])))

    # check if non test files DO NOT have match in sanitized unit test file names
    files_without_test = non_test_files - {file[1] for file in test_files}

    # check IF sanitized unit test file names have match in non test files
    files_with_test = set()
    tests_without_file = set()
    for file_tuple in test_files:
        if file_tuple[1] in non_test_files:
            # create tuple with normal file name and unit test file name
            files_with_test.add((file_tuple[1], file_tuple[0]))
        else:
            tests_without_file.add(file_tuple[0])

    return files_without_test, files_with_test, tests_without_file

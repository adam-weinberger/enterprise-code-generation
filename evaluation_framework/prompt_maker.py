import os
import random
from typing import List, Tuple

from javalang import javalang
import pandas as pd

from config import config_dict, logger

"""Adam: Overall
The code is split up well into modular functions. The ability to make three types of prompts is nice
How do these functions work together? Is choose_count_random_line_eval_df the top level function? What is the purpose of per_file_diverse_eval_df?
Add type hints to the functions
"""

def get_relevant_line_indices(lines,num_label_lines=1,min_lines_above=1,min_lines_below=1):
    """
    Takes lines of code and creates a list of line indices for lines that are not used for whitespace, comments, imports, or packages. 
    """

    # remove empty lines, single line comments, import statements, packages
    relevant_lines_indices_with_mlcomments = [index for index, line in enumerate(lines) if line.strip() != '']
    relevant_lines_indices_with_mlcomments = [index for index in relevant_lines_indices_with_mlcomments if not lines[index].strip().startswith('//')]
    relevant_lines_indices_with_mlcomments = [index for index in relevant_lines_indices_with_mlcomments if not lines[index].strip().startswith('#')]
    relevant_lines_indices_with_mlcomments = [index for index in relevant_lines_indices_with_mlcomments if not (lines[index].strip().startswith('import') or lines[index].strip().startswith('package'))]

    # To handle multiline comments, we can use a flag
    in_multiline_comment = False
    relevant_line_indices = []

    for index in relevant_lines_indices_with_mlcomments:
        line = lines[index]

        if line.strip().startswith('/*'):
            in_multiline_comment = True

        if not in_multiline_comment:
            relevant_line_indices.append(index)

        if line.strip().endswith('*/'):
            in_multiline_comment = False
    
    # can only choose a line for a prompt if there are enough lines of context above and below
    # Adam: Why? any line should be possible to be chosen for prompt.
    trimmed_relevant_line_indices = relevant_line_indices[min_lines_above+1:- (num_label_lines + min_lines_below)]

    return trimmed_relevant_line_indices

def get_relevant_token_indices(lines, num_label_tokens=1, min_tokens_before=1, min_tokens_after=1):
    """
    Takes lines of code and creates a list of token indices for tokens that are not used for whitespace, comments, imports, or packages. 
    """

    code = "".join(lines)
    tokens = list(javalang.tokenizer.tokenize(code))

    # Adam: the logic for deciding what is a comment here is different than in the get_relevant_line_indices function
    # Get the indices of tokens that are not whitespace or comments.
    relevant_token_indices_with_mlcomments = [index for index, token in enumerate(tokens) if token.value.strip() != '' and not ('Comment' in token.__class__.__name__)]

    # Filter out 'import' and 'package' tokens.
    relevant_token_indices = [index for index in relevant_token_indices_with_mlcomments if tokens[index].value not in ['import', 'package']]

    # Trimming tokens
    trimmed_relevant_token_indices = relevant_token_indices[min_tokens_before:- (num_label_tokens + min_tokens_after)]

    return trimmed_relevant_token_indices


def find_method_in_file(lines, file_path):
    """
    Given a method name, return the first line number and the length of the body the method.
    """

    code = "".join(lines)
    tree = javalang.parse.parse(code)

    for path, node in tree.filter(javalang.tree.MethodDeclaration or javalang.tree.ConstructorDeclaration):

        start_line = node.position.line - 1  # javalang counts lines starting with 1 not 0
        method_content = "".join(lines[start_line:])
        method_tokens = list(javalang.tokenizer.tokenize(method_content))
        body_start_line, body_line_count = None, None

        stack = []
        for token in method_tokens:
            if token.value == '{':
                stack.append('{')
                if not body_start_line:
                    body_start_line = start_line + token.position.line  # assume that the function starts on the line after the opening { (+1) but also javalang token.position.line starts at index 1 not 0 (-1) so they cancel out

            elif token.value == '}':
                stack.pop()
                if not stack:
                    body_line_count = (start_line + token.position.line) - body_start_line - 1  # assume that the last line is only a } and do not count it
                    break

        yield (file_path, node.name, body_start_line, body_line_count)
    return None


def create_prompt(prefix_lines,suffix_lines,infill=True,max_lines_above=50,max_lines_below=50):
    """
    Takes the prefix_lines and suffix_lines and uses them to consruct the prompt. Defines prompt style. Limits prompt size. 
    """

    # Limit the number of lines in the prefix and suffix to a maximum number of lines. 
    if len(prefix_lines) > max_lines_above:
        prefix_lines = prefix_lines[-max_lines_above:]
        
    if len(suffix_lines) > max_lines_below:
        suffix_lines = suffix_lines[:max_lines_below]

    if infill == True:
        prompt = f'<PRE> {"".join(prefix_lines)} <SUF>{"".join(suffix_lines)} <MID>'
    else:
        prompt = "".join(prefix_lines)

    return prompt

def line_removal_prompt(lines,line_to_remove_index=None,num_label_lines=1,infill=True,max_lines_above=50,max_lines_below=50):
    """
    Given lines of code, will remove either a single line or multiple lines in order to create infill prompts. 
    """
    
    if line_to_remove_index == None:
        relevant_line_indices = get_relevant_line_indices(lines)
        line_to_remove_index = random.choice(relevant_line_indices)

    last_line_to_remove_index = line_to_remove_index + num_label_lines
    
    prefix_lines = lines[:line_to_remove_index]
    suffix_lines = lines[last_line_to_remove_index:]

    if infill == True:
        label = "".join(lines[line_to_remove_index:last_line_to_remove_index])
    else:
        label = "".join(lines[line_to_remove_index:])

    prompt = create_prompt(prefix_lines, suffix_lines, infill=infill,  max_lines_above=max_lines_above, max_lines_below=max_lines_below)

    return [prompt, label, line_to_remove_index + 1, last_line_to_remove_index]

def token_removal_prompt(lines,starting_token_index = None,num_label_tokens=1,infill=True,max_lines_above=50,max_lines_below=50):
    """
    Given lines of code, will remove a random set of consecutive tokens to create an infill prompt. 
    """

    if starting_token_index == None:
        relevant_token_indices = get_relevant_token_indices(lines,num_label_tokens=num_label_tokens)
        starting_token_index = random.choice(relevant_token_indices)
    
    code = "".join(lines)
    tokens = list(javalang.tokenizer.tokenize(code))

    start_position = tokens[starting_token_index].position[1] - 1  # this gets the column start of the token
    end_position = tokens[starting_token_index + num_label_tokens - 1].position[1] - 1 + len(tokens[starting_token_index + num_label_tokens - 1].value)  # this gets the column end of the last token
    start_line = tokens[starting_token_index].position[0]
    end_line = tokens[starting_token_index + num_label_tokens - 1].position[0]
    
    # List of lines before start line and the piece of line before the label. 
    prefix_lines = lines[:tokens[starting_token_index].position[0] - 1] + [lines[tokens[starting_token_index].position[0] - 1][:start_position]]
    # List of the piece of line after the label and lines after end line. 
    suffix_lines = [lines[tokens[starting_token_index + num_label_tokens - 1].position[0] - 1][end_position:]] + lines[tokens[starting_token_index + num_label_tokens - 1].position[0]:]

    prompt = create_prompt(prefix_lines, suffix_lines, infill=infill,  max_lines_above=max_lines_above, max_lines_below=max_lines_below)

    if infill == True:
        label = code[len("".join(prefix_lines)):len(code)-len("".join(suffix_lines))]
    else:
        label= code[len("".join(prefix_lines)):]
    
    return [prompt, label, start_line, end_line]


def method_declaration_prompt(lines,method_name,start_line,end_line,infill=True,max_lines_above=50,max_lines_below=50):
    """
    Creates a method declaration prompt/label and a corresponding row. 
    """

    if start_line != None and end_line != None:
        
        prefix_lines = lines[:start_line]
        suffix_lines = lines[end_line:]
        prompt = create_prompt(prefix_lines, suffix_lines, infill=infill,  max_lines_above=max_lines_above, max_lines_below=max_lines_below)

        if infill == True:
            label = "".join(lines[start_line:end_line])
        else:
            label = "".join(lines[start_line:])

    return [method_name, prompt, label, start_line, end_line]

def make_prompts(
        output_directory: str,
        num_label_lines: int,
        lines_prompts_count: int,
        num_label_tokens: int,
        tokens_prompts_count: int,
        methods_prompts_count: int,
        infill: bool = True,
        max_lines_above: int = 50,
        max_lines_below: int = 50) -> Tuple:
    """
    Given a folder of java files, returns a dictionary with three dataframes corresponding to three different kinds of prompts. 
    """
    
    logger.info("Make prompts")

    skipped_files = []

    test_folder_path = os.path.join(output_directory,'test/')
    mapper_path = os.path.join(output_directory, 'file_mappings_test.csv')
    path_map_df = pd.read_csv(mapper_path)

    logger.info("Getting relevant line indices for prompt-making")
    all_relevant_line_indices = []
    all_relevant_token_indices = []
    all_method_bodies = []

    # Iterate through the files in the folder
    for root, dirs, files in os.walk(test_folder_path):
        for file in files:
            if file.endswith('.java'):
                file_path = os.path.join(root, file)

                with open(file_path, 'r') as f:
                    lines = f.readlines()

                # get indices of lines and tokens that correspond to normal code and are not empty or comments or imports
                these_relevant_line_indices = [(file_path, num) for num in get_relevant_line_indices(lines, num_label_lines)]
                all_relevant_line_indices.extend(these_relevant_line_indices)
                try:
                    these_relevant_token_indices = [(file_path, num) for num in get_relevant_token_indices(lines, num_label_tokens)]
                    all_relevant_token_indices.extend(these_relevant_token_indices)
                except Exception as e:
                    skipped_files.append((file_path, "these_relevant_token_indices"))

                try:
                    all_method_bodies.extend(find_method_in_file(lines, file_path))
                except Exception as e:
                    skipped_files.append((file_path, "all_relevant_method_bodies"))

    num_skips = len(skipped_files)
    if num_skips > 0:
        logger.error(f"Error processing {num_skips} files. See skipped_files.csv for details.")
    
    random.seed(config_dict['seed'])
    if lines_prompts_count < len(all_relevant_line_indices):
        line_indices_for_lines_prompt_making = random.sample(all_relevant_line_indices, lines_prompts_count)
    else:
        logger.warning(f"Not enough relevant lines for prompt-making. Using all {len(all_relevant_line_indices)} relevant lines.")
        line_indices_for_lines_prompt_making = all_relevant_line_indices
    if tokens_prompts_count < len(all_relevant_token_indices):
        token_indices_for_tokens_prompt_making = random.sample(all_relevant_token_indices, tokens_prompts_count)
    else:
        logger.warning(f"Not enough relevant tokens for prompt-making. Using all {len(all_relevant_token_indices)} relevant tokens.")
        token_indices_for_tokens_prompt_making = all_relevant_token_indices
    if methods_prompts_count < len(all_method_bodies):
        body_indices_for_method_prompt_making = random.sample(all_method_bodies, methods_prompts_count)
    else:
        logger.warning(f"Not enough methods for prompt-making. Using all {len(all_method_bodies)} methods.")
        body_indices_for_method_prompt_making = all_method_bodies

    logger.info("Creating method declaration body infill prompts")
    methods_df_list = []
    for path, method, body_start_line, body_line_count in body_indices_for_method_prompt_making:
        with open(path, 'r') as f:
            lines = f.readlines()

        try:
            methods_df_row = line_removal_prompt(lines, body_start_line, body_line_count, infill, max_lines_above, max_lines_below)

            if methods_df_row != []:
                old_path = path_map_df.loc[path_map_df['new_path'] == path, 'old_path'].iloc[0]
                methods_df_list.append([path, old_path, method]+methods_df_row)
        except Exception as e:
            logger.error(f"Error processing {path} for method_removal_prompts: {e}")
            skipped_files.append((path, "method_removal_prompts"))

    logger.info("Creating token infill prompts")
    tokens_df_list = []
    for path, starting_token_index in token_indices_for_tokens_prompt_making:

        with open(path, 'r') as f:
            lines = f.readlines()

        try:
            tokens_df_row = token_removal_prompt(
                lines,
                starting_token_index,
                num_label_tokens,
                infill,
                max_lines_above,
                max_lines_below
            )

            if tokens_df_row != []:
                old_path = path_map_df.loc[path_map_df['new_path'] == path, 'old_path'].iloc[0]
                tokens_df_list.append([path,old_path]+tokens_df_row)
        except Exception as e:
            logger.error(f"Error processing {path} for token_removal_prompts: {e}")
            skipped_files.append((path, "token_removal_prompts"))

    logger.info("Creating line infill prompts")
    lines_df_list = []
    for path, line_to_remove_index in line_indices_for_lines_prompt_making:

        with open(path, 'r') as f:
            lines = f.readlines()

        try:
            lines_df_row = line_removal_prompt(
                lines,
                line_to_remove_index,
                num_label_lines,
                infill,
                max_lines_above,
                max_lines_below
            )

            if lines_df_row != []:
                old_path = path_map_df.loc[path_map_df['new_path'] == path, 'old_path'].iloc[0]
                lines_df_list.append([path,old_path]+lines_df_row)
        except Exception as e:
            logger.error(f"Error processing {path} for line_removal_prompts: {e}")
            skipped_files.append((path, "line_removal_prompts"))

    df_dict = {
        'lines_df': pd.DataFrame(lines_df_list, columns=["path", "path_in_snap_repo","prompt", "label", "line_to_remove_index", "last_line_to_remove_index"]),
        'tokens_df': pd.DataFrame(tokens_df_list, columns=["path", "path_in_snap_repo", "prompt", "label", "start_line", "end_line"]),
        'methods_df': pd.DataFrame(methods_df_list, columns=["path", "path_in_snap_repo", "method", "prompt", "label", "start_line", "end_line"])
        }

    output_directory = os.path.join(config_dict["output_directory"], 'prompts/')

    os.makedirs(output_directory, exist_ok=True)

    for df in df_dict.items():
        df[1].to_csv(os.path.join(output_directory, f'{df[0]}.csv'), index=False)

    skipped_files_df = pd.DataFrame(skipped_files, columns=['file_path', 'reason']) \
                        .merge(path_map_df, left_on='file_path', right_on='new_path') \
                        .filter(items=['old_path','reason'])
    
    skipped_files_csv_path = os.path.join(output_directory, 'skipped_files.csv')
    skipped_files_df.to_csv(skipped_files_csv_path, index=False)

    pd.set_option('max_colwidth', 500)

    return df_dict, skipped_files

import os
from pathlib import Path
import random
import shutil
from typing import List, Tuple
import pandas as pd

from config import logger


def split_train_test(
    code_base_directory: str, train_ratio=0.8, output_directory=None, seed=0, file_mapping_name="file_mappings", split_strategy="random"
):
    """
    Given a directory (TODO repository, zip, url)
    Scan it for .java files (TODO a specified file type)
    Randomly split the .java files into train and test (make modular for a different splitting method)
    Generate a directory for the train and test files respectively (TODO zip)

    :param code_base_directory: The directory containing .java files.
    :param train_ratio: The ratio of files to include in the training set (default is 0.8).
    :param output_directory: The base directory where train and test directories will be created (default is None).
    :param seed: random seed for splitting train and test (default is 0)
    """
    logger.info("Split train test")

    
    if split_strategy == "random":
        java_files = _find_java_files(code_base_directory)
        train_files, test_files = _split_train_test(java_files, train_ratio, seed)
    elif split_strategy == "by_module":
        train_files, test_files = split_train_test_per_module(code_base_directory, train_ratio, seed)

    logger.info("Copying train and test files to output directory")
    _copy_files_to_directory(train_files, code_base_directory, output_directory, file_mapping_name, "train")
    _copy_files_to_directory(test_files, code_base_directory, output_directory, file_mapping_name, "test")


def _find_java_files(directory: str) -> List[Tuple[str, str]]:
    """
    Search for Java (.java) files within the specified directory and its subdirectories.

    :param directory: The directory to start the search from.
    :return: A list of paths to Java files found.
    """
    logger.info("Finding Java files in %s", directory)
    java_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".java") and "test" not in file.lower():
                java_files.append((root, file))
    return java_files


def _split_train_test(
    java_files: List, train_ratio=0.8, seed=0
) -> Tuple[List[str], List[str]]:
    """
    Randomly split a list of Java files into training and test sets.

    :param java_files: A list of file paths to Java (.java) files to be split.
    :param train_ratio: The ratio of files to include in the training set (default is 0.8).
    :param seed: Seed for random number generation to ensure reproducibility (default is 0).
    :return: A tuple containing two lists: the training files and the test files.
    """
    logger.info("Splitting Java files into train and test sets")
    
    random.seed(seed)
    random.shuffle(java_files)

    split_index = int(train_ratio * len(java_files))

    train_files = java_files[:split_index]
    test_files = java_files[split_index:]

    return train_files, test_files


def split_train_test_per_module(
    directory: str, java_files: List[Tuple[str, str]], seed=0
):
    """
    For each module, which can be identified by removing the directory from
    the java_file and then taking the top level folder,
    put one file in the test set and the rest in the training set
    """

    # Seed the random number generator for reproducibility.
    random.seed(seed)

    # Create a dictionary to group Java files by their respective modules.
    module_files = {}
    for root, filename in java_files:
        module = root.replace(directory, "")
        module = Path(module).parts[0]
        if module not in module_files:
            module_files[module] = []
        module_files[module].append(filename)

    # Initialize the training and test sets.
    train_set = []
    test_set = []

    # Iterate through modules and split files into training and test sets.
    for module, files in module_files.items():
        # Randomly shuffle the files.
        random.shuffle(files)

        # Take one file for the test set, and the rest for the training set.
        test_file = files.pop()

        # Add the test file to the test set.
        test_set.append((module, test_file))

        # Add the remaining files to the training set.
        train_set.extend([(module, file) for file in files])

    return train_set, test_set


def _copy_files_to_directory(
    files: List[str],
    source_directory: str,
    destination_parent_directory: str,
    file_mapping_name: str,
    new_directory_name: str,
):
    """
    Copy a list of files to a destination directory, preserving their relative structure.

    :param files: A list of file paths to be copied.
    :param source_directory: The base directory where the source files are located.
    :param destination_directory: The base directory where the copied files will be placed.
    :param new_directory_name: (Optional) The name of the subdirectory within the destination directory to copy the files to.
    """

    if destination_parent_directory is None:
        destination_parent_directory = Path.home()

    # create directory from scratch each time
    destination_directory = os.path.join(destination_parent_directory, new_directory_name)
    if os.path.exists(destination_directory):
        shutil.rmtree(destination_directory)
    os.makedirs(destination_directory)
    logger.debug(f"Copying to {destination_directory}")

    # copy each file into the destination directory ignoring their relative path within the code base
    # store the relative path in a separate csv file
    source_destination_mappings =[]
    id = 1
    for root, file in files:
        id_file = str(id) + '_' + file
        id += 1
        destination_path = os.path.join(destination_directory, id_file)
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        source_path = os.path.join(root, file)
        shutil.copy(source_path, destination_path)

        parts = source_path.split(source_directory, 1)
        relative_source_path = source_directory + parts[1]
        source_destination_mappings.append((relative_source_path, destination_path))

    df = pd.DataFrame(source_destination_mappings, columns=['old_path', 'new_path'])
    mappings_filepath = os.path.join(destination_parent_directory, f"{file_mapping_name}_{new_directory_name}.csv")
    df.to_csv(mappings_filepath, index=False)
    
    # zip the files for processing by refact
    zip_file = f"{new_directory_name}.zip"
    os.system(f"zip -r -q {zip_file} {destination_directory}")

    if os.path.exists(os.path.join(destination_parent_directory, f"{new_directory_name}.zip")):
        os.remove(os.path.join(destination_parent_directory, f"{new_directory_name}.zip"))
    shutil.move(zip_file, destination_parent_directory)

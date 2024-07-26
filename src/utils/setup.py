# LOAD MODULES
# Standard library
import os
from typing import Any, Dict, Tuple, List
import csv
import json
import re

# Third party
import yaml

# CUSTOM FUNCTIONS
def load_config(file_path: str) -> Dict[str, Any]:
    """
    Load a configuration file from a file path.

    Parameters:
    file_path (str): The file path of the configuration file.

    Returns:
    dict: The loaded configuration as a dictionary.
    """
    with open(file_path) as file:
        config = yaml.safe_load(file)
    
    return config

def check_create_dir(dir_path: str) -> None:
    """
    Check if a directory exists and create it if it does not.

    Parameters:
    dir_path (str): The path of the directory to check.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def check_create_csv(
    file_path: str,
    header: Tuple,
) -> None:
    """
    Check if a csv file exists and create it if it does not.

    Parameters:
    file_path (str): The path of the file to check.
    header (tuple): The header of the file to create as a tuple of strings.
    """
    with open(file_path, "a") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        
        # Write header
        if csvfile.tell() == 0:
            writer.writeheader()

def get_rows(
    file_path: str,
) -> List:
    """
    Get the rows of a csv file.

    Parameters:
    file_path (str): The path of the file to check.

    Returns:
    list: A list of tuples representing the rows.
    """
    rows = []
    with open(file_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        
        for row in reader:
            parsed_row = []
            for item in row:
                if item.isdigit():
                    parsed_row.append(int(item))
                elif item.replace('.', '', 1).isdigit():
                    parsed_row.append(float(item))
                else:
                    parsed_row.append(item)
            rows.append(tuple(parsed_row))
    
    return rows

def add_row(
    file_path: str,
    row: Tuple,
) -> None:
    """
    Add a row to a csv file.

    Parameters:
    file_path (str): The path of the file to check.
    row (Tuple): A tuple containing the row to add.
    """
    with open(file_path, "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row)

def add_dict(
    file_path: str,
    row: Dict,
) -> None:
    """
    Add a row to a csv file.
    If the file is empty, the header will be written first.

    Parameters:
    file_path (str): The path of the file to check.
    row (Dict): A dictionary containing the row to add.
    """
    with open(file_path, "a", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row.keys())
        
        # Write header if file is empty
        if csvfile.tell() == 0:
            writer.writeheader()
        
        # Write row
        writer.writerow(row)
        
def dict_to_filename(d: dict) -> str:
    """
    Converts a dictionary to a filename.

    Parameters:
    d (dict): The input dictionary.

    Returns:
    str: The filename.
    """
    filename = json.dumps(d)
    filename = re.sub(r'[\\/*?:"<>|]', "", filename)  # Remove characters not allowed in filenames
    return filename
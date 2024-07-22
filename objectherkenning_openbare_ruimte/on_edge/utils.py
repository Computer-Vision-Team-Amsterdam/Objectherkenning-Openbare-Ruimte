import logging
import os
import pathlib
import shutil
import time
from functools import wraps

logger = logging.getLogger(__name__)


def get_frame_metadata_csv_file_paths(root_folder):
    csvs = []
    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            if (
                filename.endswith("csv")
                and filename != "runs.csv"
                and filename != "system_metrics.csv"
            ):
                filepath = os.path.join(foldername, filename)
                csvs.append(filepath)
    return csvs


def count_files_in_folder_tree(root_folder: pathlib.Path, file_type: str):
    """
    Counts how many files of a specific type are in a folder and all the subfolders.
    The type is for example: "csv", "jpeg", ...

    Parameters
    ----------
    root_folder : pathlib.Path
        Root folder
    file_type : str
        Type of file to filter by

    Returns
    -------
    int
        Count of how many files
    """
    count = 0
    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith(file_type):
                count += 1
    return count


def get_img_name_from_csv_row(csv_path, row):
    csv_path_split = csv_path.stem.split(sep="-", maxsplit=1)
    img_name = f"0-{csv_path_split[1]}-{row[1]}.jpg"
    return img_name


def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.debug(f"Starting {func.__name__}...")
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Finished {func.__name__} in {duration:.4f} seconds.")
        return result

    return wrapper


def move_file(file_path, output_file_path):
    try:
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        shutil.move(file_path, output_file_path)
        logger.info(f"{file_path} has been moved to {output_file_path}.")
    except FileNotFoundError:
        logger.error(f"{file_path} does not exist.")
    except Exception as e:
        logger.error(f"Failed to move file '{file_path}': {str(e)}")
        raise Exception(f"Failed to move file '{file_path}': {e}")

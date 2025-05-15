import argparse
import ast
from datetime import datetime
from typing import Any, List

from databricks.sdk.runtime import *  # noqa: F403

from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.table_manager import (
    TableManager,
)


def setup_arg_parser(prog: str = __name__) -> argparse.ArgumentParser:
    """
    Setup an ArgumentParser for the command line parameters / job parameters.
    """
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument(
        "--stadsdelen", type=str, default="", help="\"['name1', 'name2', ...]\""
    )
    parser.add_argument(
        "--send_limits",
        type=str,
        default="",
        help='"[{2: x, 3: y, 4: z}, {2: x2, 3: y2, 4: z2}, ...]"',
    )
    return parser


def parse_task_args_to_settings(
    settings: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    """
    Parse the command line arguments and update the active_task settings.

    Parameters
    ----------
    settings: dict[str, Any]
        Full databricks config settings.
    args: argparse.Namespace
        Command line arguments

    Returns
    -------
    Updated settings dict
    """

    def _parse_stadsdelen_arg(arg_str: str) -> List[str]:
        _stadsdelen = ast.literal_eval(arg_str)
        if isinstance(_stadsdelen, str):
            _stadsdelen = [_stadsdelen]
        return [_s.strip().capitalize() for _s in _stadsdelen]

    def _parse_send_limits_arg(arg_str: str) -> List[dict[int, int]]:
        _send_limits = ast.literal_eval(arg_str)
        if isinstance(_send_limits, dict):
            _send_limits = [_send_limits]
        return _send_limits

    if args.stadsdelen:
        stadsdelen = _parse_stadsdelen_arg(args.stadsdelen)
    else:
        print("Using default stadsdelen.")
        stadsdelen = settings["job_config"]["active_task"].keys()

    if args.send_limits:
        send_limits = _parse_send_limits_arg(args.send_limits)
    else:
        print("Using default send limits.")
        send_limits = None

    if send_limits and not stadsdelen:
        raise ValueError(
            "Must provide parameter `stadsdelen` if `send_limits` are given."
        )
    if (stadsdelen and send_limits) and not (len(stadsdelen) == len(send_limits)):
        raise ValueError(
            f"Argument number mismatch: {len(stadsdelen)} stadsdelen with {len(send_limits)} send limits."
        )

    active_tasks = {}

    for i, stadsdeel in enumerate(stadsdelen):
        if send_limits:
            active_tasks[stadsdeel] = {
                "active_object_classes": list(send_limits[i].keys()),
                "send_limit": send_limits[i],
            }
        elif stadsdeel not in settings["job_config"]["active_task"].keys():
            active_tasks[stadsdeel] = {
                "active_object_classes": [],
                "send_limit": {},
            }
        else:
            active_tasks[stadsdeel] = settings["job_config"]["active_task"][stadsdeel]

    settings["job_config"]["active_task"] = active_tasks

    return settings


def delete_file(databricks_volume_full_path):
    """
    Delete file from storage account.

    Parameters:
    databricks_volume_full_path (str): The full path of the file using the volumes. The path should be within the /Volumes/ directory in Databricks.

    """
    try:
        # if the file exists, remove it
        dbutils.fs.rm(databricks_volume_full_path)  # type: ignore[name-defined] # noqa: F821, F405
        return True
    except Exception as e:
        # Check if the error is due to the file not being found
        if "FileNotFoundException" in str(e):
            print(
                f"File {databricks_volume_full_path} does not exist, so it cannot be removed."
            )
            return False
        else:
            # If there’s another type of error, raise it
            raise RuntimeError(f"An unexpected error occurred: {e}")


def compare_dataframes(df1, df2, df1_name, df2_name):
    print(f"Comparing dataframes {df1_name} and {df2_name}.")
    print(50 * "-")

    same_count = df1.count() == df2.count()
    print(f"Same number of rows: {same_count}")

    diff1 = df1.subtract(df2)
    diff2 = df2.subtract(df1)

    if diff1.count() == 0 and diff2.count() == 0:
        print("The DataFrames have the same content.")
    else:
        print("The DataFrames differ.")

    same_schema = df1.schema == df2.schema
    if same_schema:
        print("Same schema.")
    else:
        print("\nSchemas differ. Here are the details:")
        print(f"Schema of {df1_name}:")
        df1.printSchema()
        print(f"Schema of {df2_name}:")
        df2.printSchema()
    print(50 * "-")


def unix_to_yyyy_mm_dd(unix_timestamp) -> str:
    date_time = datetime.fromtimestamp(unix_timestamp)
    return date_time.strftime("%Y-%m-%d")


def setup_tables(spark, catalog, schema):
    TableManager.spark = spark
    TableManager.catalog = catalog
    TableManager.schema = schema

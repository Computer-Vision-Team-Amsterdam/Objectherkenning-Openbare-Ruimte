import argparse
import ast
import datetime
from typing import Any, Dict, List, Optional


def setup_arg_parser(prog: str = __name__) -> argparse.ArgumentParser:
    """
    Setup an ArgumentParser for the command line parameters / job parameters.
    """
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("--date", type=str, default="", help="yyyy-mm-dd")
    parser.add_argument(
        "--manual_inspection",
        type=ast.literal_eval,
        default=False,
        help='"True" | "False" (default)',
    )
    parser.add_argument(
        "--send_limits",
        type=str,
        default="",
        help='"[{2: x, 3: y, 4: z}, {2: x2, 3: y2, 4: z2}, ...]"',
    )
    parser.add_argument("--skip_ids", type=str, default="", help='"[id1, id2, ...]"')
    parser.add_argument(
        "--stadsdelen", type=str, default="", help="\"['name1', 'name2', ...]\""
    )
    parser.add_argument(
        "-f",
        type=str,
        default="",
        help="Dummy argument to prevent errors when running pipeline step interactively.",
    )
    return parser


def parse_manual_run_arg_to_settings(
    settings: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    """
    Parse the command line arguments and update the cluster_distances setting to
    disable clustering for a manual run. Also disable annotate_detection_images.

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
    if args.manual_inspection is True:
        print(
            "Manual inspection set, image annotation and clustering will be disabled."
        )
        settings["job_config"]["annotate_detection_images"] = False

        cluster_distances: Dict[int, float] = settings["job_config"]["object_classes"][
            "cluster_distances"
        ]
        for key in cluster_distances.keys():
            cluster_distances[key] = -1  # disable clustering
        settings["job_config"]["object_classes"][
            "cluster_distances"
        ] = cluster_distances

    return settings


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

    def _parse_detection_date(arg_str: str) -> Optional[datetime.date]:
        try:
            return datetime.date.fromisoformat(arg_str)
        except ValueError as e:
            print(f"Incorrect date format, expected yyyy-mm-dd, got {arg_str}")
            raise e

    def _parse_skip_ids_arg(arg_str: str) -> List[int]:
        _skip_ids = ast.literal_eval(arg_str)
        if isinstance(_skip_ids, int):
            _skip_ids = [_skip_ids]
        return _skip_ids

    if args.send_limits and not args.stadsdelen:
        raise ValueError(
            "Must provide parameter `--stadsdelen` if `--send_limits` are given."
        )

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

    if (stadsdelen and send_limits) and not (len(stadsdelen) == len(send_limits)):
        raise ValueError(
            f"Argument number mismatch: {len(stadsdelen)} stadsdelen with {len(send_limits)} send limits."
        )

    if args.date:
        settings["job_config"]["detection_date"] = _parse_detection_date(args.date)
    else:
        settings["job_config"]["detection_date"] = None

    if args.skip_ids:
        skip_ids = _parse_skip_ids_arg(args.skip_ids)
        settings["job_config"]["skip_ids"] = skip_ids
    else:
        settings["job_config"]["skip_ids"] = []

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

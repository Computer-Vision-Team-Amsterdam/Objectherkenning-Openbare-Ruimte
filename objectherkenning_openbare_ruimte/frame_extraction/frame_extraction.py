import argparse
import os
import pathlib
import sys
from datetime import datetime

sys.path.append(os.getcwd())

from objectherkenning_openbare_ruimte.settings.settings import (  # noqa: E402
    ObjectherkenningOpenbareRuimteSettings,
)

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "config.yml")
)

ObjectherkenningOpenbareRuimteSettings.set_from_yaml(config_path)
settings = ObjectherkenningOpenbareRuimteSettings.get_settings()


def extract_frames(input_folder: str, output_folder: str):
    """
    Recursively iterates through all MP4 files in the input_folder and its subfolders, extracts frames at a given framerate, and applies distortion correction.

    All frames are extracted to the same folder, with their names reflecting the name of the source video file. In a separate folder log files will be written, one general with the executed ffmpeg commands, and individual log files for each ffmpeg operation.

    Usage:
    python frame_extraction.py --input_folder <INPUT> --output_folder <OUTPUT>

    Note:
    ffmpeg >= v4.4.3 is required

    Parameters
    ----------
    input_folder: str
        Folder containing MP4 files to be processed.
    output_folder: str
        Folder to which extracted frames and log files will be written.
    """
    fe_settings = settings["frame_extraction"]
    dc_settings = settings["distortion_correction"]

    exclude_dirs = set(fe_settings["exclude_dirs"])
    ffmpeg_filter_str = f'fps={fe_settings["fps"]},lenscorrection=cx={dc_settings["cx"]}:cy={dc_settings["cy"]}:k1={dc_settings["k1"]}:k2={dc_settings["k2"]}:i=bilinear'
    ffmpeg_args_str = (
        f"-vf '{ffmpeg_filter_str}' -q:v 1"  # -q:v 1 sets the jpeg quality
    )

    start = datetime.now()
    output_path = os.path.join(output_folder, start.strftime("%Y%m%d-%H%M%S"))
    log_path = os.path.join(
        output_folder, fe_settings["log_dir"], start.strftime("%Y%m%d-%H%M%S")
    )
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(log_path, "process_log.txt")
    with open(log_file, "w") as myfile:
        start_str = start.strftime("%d/%m/%Y %H:%M:%S")
        myfile.write(f"\n----------\nSTART RUN - {start_str}\n----------\n")

    # Walk through input_folder recursively
    for dirpath, _, filenames in os.walk(input_folder):
        if dirpath.split(sep="/")[-1] in exclude_dirs:
            continue
        for file in filenames:
            if file.endswith(".mp4"):
                file_path = os.path.join(dirpath, file)
                filename = file.rstrip(".mp4")
                ffmpeg_out_path = os.path.join(output_path, filename)
                ffmpeg_log_file = os.path.join(log_path, f"{filename}.ffmpeg_log.txt")
                ffmpeg_cmd = f"ffmpeg -i '{file_path}' {ffmpeg_args_str} '{ffmpeg_out_path}_%04d.jpg' 2> '{ffmpeg_log_file}'"

                with open(log_file, "a") as myfile:
                    myfile.write(f"{ffmpeg_cmd}\n")
                print(ffmpeg_cmd)

                os.system(ffmpeg_cmd)  # nosec B605

    with open(log_file, "a") as myfile:
        end = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        myfile.write(f"\n----------\nEND RUN - {end}\n----------\n")


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--input_folder", dest="input_folder", type=str)
    parser.add_argument("--output_folder", dest="output_folder", type=str)

    # parse args
    args = parser.parse_args()

    # return args
    return args


if __name__ == "__main__":
    args = parse_args()
    extract_frames(args.input_folder, args.output_folder)

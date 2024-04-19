import argparse
import os
import pathlib
import sys
from datetime import datetime
from subprocess import CalledProcessError, check_call  # nosec: B404

from pathvalidate import is_valid_filepath

sys.path.append(os.getcwd())

from objectherkenning_openbare_ruimte.settings.settings import (  # noqa: E402
    ObjectherkenningOpenbareRuimteSettings,
)

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "config.yml")
)

ObjectherkenningOpenbareRuimteSettings.set_from_yaml(config_path)
settings = ObjectherkenningOpenbareRuimteSettings.get_settings()


class FFMPEG_frame_extractor:
    """
    This class wraps around the ffmpeg command to facilitate frame extraction and
    distortion correction from videos.

    Checks are in place to ensure arguments are valid:
    fps, cx, cy, k1, k2 must be float
    input_file must be an existing file
    output_folder must be an existing folder
    log_file must point to a valid filename in an existing folder
    """

    def __init__(self, fps, distortion_settings):
        self.cmd = None

        cx = distortion_settings["cx"]
        cy = distortion_settings["cy"]
        k1 = distortion_settings["k1"]
        k2 = distortion_settings["k2"]

        if not isinstance(fps, float):
            print(f"Invalid type for fps: {type(fps)}")
            return
        if not isinstance(cx, float):
            print(f"Invalid type for fps: {type(fps)}")
            return
        if not isinstance(cy, float):
            print(f"Invalid type for fps: {type(fps)}")
            return
        if not isinstance(k1, float):
            print(f"Invalid type for fps: {type(fps)}")
            return
        if not isinstance(k2, float):
            print(f"Invalid type for fps: {type(fps)}")
            return

        # Prepare FFMPEG command for distortion correction and output quality
        self.filter_str = (
            f"fps={fps},lenscorrection=cx={cx}:cy={cy}:k1={k1}:k2={k2}:i=bilinear"
        )

    def create_command(self, input_file: str, output_path: str, log_file: str) -> str:
        if not os.path.isfile(input_file):
            print(f"Input file not found: {input_file}")
            return None
        if not os.path.isdir(os.path.dirname(output_path)):
            print(f"Output folder not found: {os.path.dirname(output_path)}")
            return None
        if not is_valid_filepath(output_path):
            print(f"Output file path not valid: {output_path}")
            return None
        if not os.path.isdir(os.path.dirname(log_file)):
            print(f"Log folder not found: {os.path.dirname(log_file)}")
            return None
        if not is_valid_filepath(log_file):
            print(f"Log file path not valid: {log_file}")
            return None

        self.cmd = [
            "ffmpeg",
            "-i",
            input_file,
            "-vf",
            f"'{self.filter_str}'",
            "-q:v",
            "1",  # [-q:v 1] sets output jpg quality to max
            f"{output_path}_%04d.jpg",
            "2>",
            log_file,
        ]
        return " ".join(self.cmd)

    def run(self):
        if self.cmd is None:
            print("Command not set, run 'create_cmd() first.")
        try:
            check_call(self.cmd)  # nosec: B603
        except CalledProcessError as e:
            print(e.output)


def main(input_folder: str, output_folder: str):
    """
    Recursively iterates through all MP4 files in the input_folder and its subfolders,
    extracts frames at a given framerate, and applies distortion correction.

    All frames are extracted to the same folder, with their names reflecting the name
    of the source video file. In a separate folder log files will be written, one
    general with the executed ffmpeg commands, and individual log files for each
    ffmpeg operation.

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
    exclude_dirs = set(settings["frame_extraction"]["exclude_dirs"])
    exclude_files = settings["frame_extraction"]["exclude_files"]

    log_dir = settings["frame_extraction"]["log_dir"]

    fps = settings["frame_extraction"]["fps"]
    FFMPEG = FFMPEG_frame_extractor(fps, settings["distortion_correction"])

    start = datetime.now()
    output_path = os.path.join(output_folder, start.strftime("%Y%m%d-%H%M%S"))
    log_path = os.path.join(output_folder, log_dir, start.strftime("%Y%m%d-%H%M%S"))
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)

    log_file = os.path.join(log_path, "process_log.txt")
    with open(log_file, "w") as myfile:
        start_str = start.strftime("%d/%m/%Y %H:%M:%S")
        myfile.write(f"\n----------\nSTART RUN - {start_str}\n----------\n")

    # Walk through input_folder recursively
    for dirpath, _, filenames in os.walk(input_folder):
        if dirpath.split(sep="/")[-1] in exclude_dirs:
            # Skip folders in exclude_dirs config
            continue
        for file in filenames:
            if any(excl_file in file for excl_file in exclude_files):
                # Skip files in exclude_files config
                continue
            if file.endswith(".mp4"):
                file_path = os.path.join(dirpath, file)
                filename = file.replace(".mp4", "")
                # Generate output paths for FFMPEG
                ffmpeg_out_path = os.path.join(output_path, filename)
                ffmpeg_log_file = os.path.join(log_path, f"{filename}.ffmpeg_log.txt")
                # Create full FFMPEG command
                ffmpeg_cmd = FFMPEG.create_command(
                    file_path, ffmpeg_out_path, ffmpeg_log_file
                )

                with open(log_file, "a") as myfile:
                    myfile.write(f"{ffmpeg_cmd}\n")
                print(ffmpeg_cmd)

                # Run FFMPEG command
                os.system(ffmpeg_cmd)  # nosec B605

    with open(log_file, "a") as myfile:
        end = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        myfile.write(f"\n----------\nEND RUN - {end}\n----------\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", dest="input_folder", type=str)
    parser.add_argument("--output_folder", dest="output_folder", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args.input_folder, args.output_folder)

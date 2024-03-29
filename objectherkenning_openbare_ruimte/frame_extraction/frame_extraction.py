import argparse
import os
import pathlib
from datetime import datetime

exclude_dirs = ("checkboard", "frames")
log_dir = "logs"


def main(args):
    start = datetime.now()
    output_path = os.path.join(args.output_folder, start.strftime("%Y%m%d-%H%M%S"))
    log_path = os.path.join(
        args.output_folder, log_dir, start.strftime("%Y%m%d-%H%M%S")
    )
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(log_path, "process_log.txt")
    with open(log_file, "w") as myfile:
        start_str = start.strftime("%d/%m/%Y %H:%M:%S")
        myfile.write(f"\n----------\nSTART RUN - {start_str}\n----------\n")

    for dirpath, dirnames, filenames in os.walk(args.input_folder):
        if dirpath.split(sep="/")[-1] in exclude_dirs:
            continue
        for file in filenames:
            if file.endswith(".mp4"):
                file_path = os.path.join(dirpath, file)
                filename = file.rstrip(".mp4")
                ffmpeg_out_path = os.path.join(output_path, filename)
                ffmpeg_log_file = os.path.join(log_path, f"{filename}.ffmpeg_log.txt")
                ffmpeg_cmd = f"ffmpeg -i '{file_path}' -vf 'fps=1,lenscorrection=cx=0.509:cy=0.488:k1=-0.241:k2=0.106:i=bilinear' -q:v 1 '{ffmpeg_out_path}_%04d.jpg' 2> '{ffmpeg_log_file}'"

                with open(log_file, "a") as myfile:
                    myfile.write(f"{ffmpeg_cmd}\n")
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
    main(args)

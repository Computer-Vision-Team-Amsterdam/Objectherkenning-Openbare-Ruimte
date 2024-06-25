import os


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


def get_img_name_from_csv_row(csv_path, row):
    csv_path_split = csv_path.stem.split(sep="-", maxsplit=1)
    img_name = f"0-{csv_path_split[1]}-{row[1]}.jpg"
    return img_name

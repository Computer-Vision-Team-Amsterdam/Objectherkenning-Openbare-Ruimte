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

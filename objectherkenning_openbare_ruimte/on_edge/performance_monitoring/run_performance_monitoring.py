import logging
import pathlib
import socket
from datetime import datetime
from time import sleep

import psutil
import torch

from objectherkenning_openbare_ruimte.on_edge.utils import count_files_in_folder_tree
from objectherkenning_openbare_ruimte.settings.luna_logging import setup_luna_logging
from objectherkenning_openbare_ruimte.settings.settings import (
    ObjectherkenningOpenbareRuimteSettings,
)


def internet(host="8.8.8.8", port=53, timeout=3):
    """
    Host: 8.8.8.8 (google-public-dns-a.google.com)
    OpenPort: 53/tcp
    Service: domain (DNS/TCP)
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error as ex:
        logger.error(ex)
        return False


if __name__ == "__main__":
    settings = ObjectherkenningOpenbareRuimteSettings.set_from_yaml("config.yml")
    logging_file_path = f"{settings['logging']['luna_logs_dir']}/performance_monitoring/{datetime.now()}.txt"
    setup_luna_logging(settings["logging"], logging_file_path)
    logger = logging.getLogger("performance_monitoring")
    images_folder = pathlib.Path(settings["detection_pipeline"]["images_path"])
    detections_folder = pathlib.Path(settings["detection_pipeline"]["detections_path"])
    metadata_folder = pathlib.Path(settings["data_delivery_pipeline"]["metadata_path"])

    logger.info("Performance monitor is running. It will start providing updates soon.")
    while True:
        gpu_device_name = (
            torch.cuda.get_device_name()
            if torch.cuda.is_available()
            else "GPU not available"
        )
        ram_load = psutil.virtual_memory().percent
        cpu_load = psutil.cpu_percent()
        logger.info(
            f"system_status: [internet: {internet()}, cpu: {cpu_load}, ram: {ram_load}, gpu_device_name: {gpu_device_name}]"
        )
        logger.info(
            f"folder_status: ["
            f"CSVs in images folder: {count_files_in_folder_tree(images_folder, 'csv') - 2}, "
            f"JPGs in images folder: {count_files_in_folder_tree(images_folder, 'jpg')}, "
            f"CSVs in detections folder: {count_files_in_folder_tree(detections_folder, 'csv')}, "
            f"JPGs in detections folder: {count_files_in_folder_tree(detections_folder, 'jpg')}, "
            f"CSVs in metadata folder: {count_files_in_folder_tree(metadata_folder, 'csv')}"
            f"]"
        )
        sleep(30.0)

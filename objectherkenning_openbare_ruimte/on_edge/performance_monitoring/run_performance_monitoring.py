import logging
import socket
from datetime import datetime
from time import sleep

import psutil
import torch

from objectherkenning_openbare_ruimte.settings.luna_logging import setup_luna_logging
from objectherkenning_openbare_ruimte.settings.settings import (
    ObjectherkenningOpenbareRuimteSettings,
)

# from jtop import jtop


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


# def jetson():
#     with jtop() as jetson:
#         if jetson.ok():
#             logger.info(jetson.stats)
#         else:
#             logger.error("Jetson not OK")


if __name__ == "__main__":
    settings = ObjectherkenningOpenbareRuimteSettings.set_from_yaml("config.yml")
    logging_file_path = f"{settings['logging']['luna_logs_dir']}/performance_monitoring/{datetime.now()}.txt"
    setup_luna_logging(settings["logging"], logging_file_path)
    logger = logging.getLogger("performance_monitoring")
    logger.info("Performance monitor is running. It will start providing updates soon.")
    first_loop = True
    while True:
        if torch.cuda.is_available():
            gpu_status = torch.cuda.get_device_name()
        else:
            gpu_status = "GPU not available"
        ram_load = psutil.virtual_memory().percent
        cpu_load = psutil.cpu_percent()
        logger.info(
            f"system_status: [internet: {internet()}, cpu: {cpu_load}, ram: {ram_load}, gpu: {gpu_status}]"
        )
        sleep(30.0)

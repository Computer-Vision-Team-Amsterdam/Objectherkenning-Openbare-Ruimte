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
    logger.info("Performance monitor is running. It will start providing updates soon.")
    first_loop = True
    while True:
        try:
            (gpu_free, gpu_total) = torch.cuda.mem_get_info()
            vram_load = ((gpu_total - gpu_free) / gpu_total) * 100
            gpu_load = torch.cuda.utilization()
        except Exception as e:
            if first_loop:
                logger.warning(f"No GPU available: {e}")
            vram_load = gpu_load = 0
        ram_load = psutil.virtual_memory().percent
        cpu_load = psutil.cpu_percent()
        logger.info(
            f"system_status: [internet: {internet()}, cpu: {cpu_load}, ram: {ram_load}, gpu: {gpu_load}, "
            f"vram: {vram_load:.1f}]"
        )
        sleep(1.0)

from time import sleep

import psutil
import torch

if __name__ == "__main__":
    print("Performance monitor is running. It will start providing updates soon.")
    while True:
        try:
            (gpu_free, gpu_total) = torch.cuda.mem_get_info()
            vram_load = ((gpu_total - gpu_free) / gpu_total) * 100
            gpu_load = torch.cuda.utilization()
        except Exception as e:
            print(f"No GPU available: {e}")
            vram_load = gpu_load = 0
        ram_load = psutil.virtual_memory().percent
        cpu_load = psutil.cpu_percent()
        print(
            f"system_status: [cpu: {cpu_load}, ram: {ram_load}, gpu: {gpu_load}, vram: {vram_load:.1f}]"
        )
        sleep(1.0)

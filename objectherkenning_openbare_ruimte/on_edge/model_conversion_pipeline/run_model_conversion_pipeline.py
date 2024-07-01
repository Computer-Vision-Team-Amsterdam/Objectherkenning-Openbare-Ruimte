import logging
import os
import traceback
from datetime import datetime

from ultralytics import YOLO

from objectherkenning_openbare_ruimte.settings.luna_logging import setup_luna_logging
from objectherkenning_openbare_ruimte.settings.settings import (
    ObjectherkenningOpenbareRuimteSettings,
)


def run_model_conversion():
    settings = ObjectherkenningOpenbareRuimteSettings.set_from_yaml("config.yml")
    logging_file_path = f"{settings['logging']['luna_logs_dir']}/model_conversion_pipeline/{datetime.now()}.txt"
    setup_luna_logging(settings["logging"], logging_file_path)
    logger = logging.getLogger("model_conversion_pipeline")

    pretrained_model_path = settings["detection_pipeline"]["pretrained_model_path"]
    model_name = settings["detection_pipeline"]["model_name"]

    if model_name.rsplit(sep=".", maxsplit=1)[-1] == "pt":
        logger.info("Required model is a Torch model, no conversion needed.")
        return

    if os.path.isfile(os.path.join(pretrained_model_path, model_name)):
        logger.info("Converted model already exists.")
        return

    model_src_name = model_name.rsplit(sep=".", maxsplit=1)[0] + ".pt"
    if not os.path.isfile(os.path.join(pretrained_model_path, model_src_name)):
        logger.error(
            f"Cannot convert model {model_name} because {model_src_name} is not found."
        )
        return

    logger.info(f"Converting {model_src_name} to {model_name}..")

    try:
        model = YOLO(os.path.join(pretrained_model_path, model_src_name))
        model.export(
            format="engine",
            half=True,
            imgsz=settings["detection_pipeline"]["output_image_size"],
            workspace=4.0,
            batch=1,
        )
    except Exception as e:
        logger.info(f"Exception occurred in model conversion: {e}")
        logger.debug(traceback.format_exc())


if __name__ == "__main__":
    run_model_conversion()

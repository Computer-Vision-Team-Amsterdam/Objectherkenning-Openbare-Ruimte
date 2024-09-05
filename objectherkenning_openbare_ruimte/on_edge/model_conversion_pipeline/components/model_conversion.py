import logging
import os
import pathlib
import shutil
from typing import Tuple, Union

import psutil
from ultralytics import YOLO

logger = logging.getLogger("model_conversion_pipeline")


def _convert_model_to_trt(
    model_path: Union[str, os.PathLike], image_size: Tuple[int, int], batch: int = 1
) -> str:
    if psutil.virtual_memory().total > 8.0 * 1.073742e9:
        workspace = 4.0
    else:
        workspace = 2.0

    logger.debug(f"Converting model using workspace={workspace}.")
    model = YOLO(model_path, task="detect")
    model.export(
        format="engine",
        half=True,
        imgsz=image_size,
        workspace=workspace,
        batch=batch,
    )
    model_path = pathlib.Path(model_path)
    return os.path.join(model_path.parent, model_path.stem + ".engine")


def run_model_conversion(
    pretrained_model_path: str,
    model_name: str,
    image_size: Tuple[int, int],
    model_save_path: str,
    overwrite_if_exists: bool = False,
):

    model_type = model_name.rsplit(sep=".", maxsplit=1)[-1]
    if model_type == "pt":
        logger.info("Required model is a Torch model, no conversion needed.")
        return

    if model_type != "engine":
        logger.info(f"Unknown model type: {model_type}.")
        return

    if (
        os.path.isfile(os.path.join(pretrained_model_path, model_name))
        and not overwrite_if_exists
    ):
        logger.info(
            "Converted model already exists. Set overwrite_if_exists=True to overwrite existing model."
        )
        return

    model_src_name = model_name.rsplit(sep=".", maxsplit=1)[0] + ".pt"
    if not os.path.isfile(os.path.join(pretrained_model_path, model_src_name)):
        logger.error(
            f"Cannot convert model {model_name} because {model_src_name} is not found."
        )
        return

    logger.info(f"Converting {model_src_name} to {model_name}..")
    trt_path = _convert_model_to_trt(
        model_path=os.path.join(pretrained_model_path, model_src_name),
        image_size=image_size,
    )
    if model_save_path != pretrained_model_path:
        shutil.copy2(trt_path, model_save_path)
    logger.info(f"Model converted and stored in {model_save_path}/{model_name}")

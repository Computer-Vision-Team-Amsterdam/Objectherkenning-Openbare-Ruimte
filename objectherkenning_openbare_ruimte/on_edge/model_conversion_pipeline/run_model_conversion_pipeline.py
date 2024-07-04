import logging
from datetime import datetime

from objectherkenning_openbare_ruimte.on_edge.model_conversion_pipeline.components.model_conversion import (
    run_model_conversion,
)
from objectherkenning_openbare_ruimte.settings.luna_logging import setup_luna_logging
from objectherkenning_openbare_ruimte.settings.settings import (
    ObjectherkenningOpenbareRuimteSettings,
)

if __name__ == "__main__":
    settings = ObjectherkenningOpenbareRuimteSettings.set_from_yaml("config.yml")
    logging_file_path = f"{settings['logging']['luna_logs_dir']}/model_conversion_pipeline/{datetime.now()}.txt"
    setup_luna_logging(settings["logging"], logging_file_path)
    logger = logging.getLogger("model_conversion_pipeline")

    run_model_conversion(
        pretrained_model_path=settings["detection_pipeline"]["pretrained_model_path"],
        model_name=settings["detection_pipeline"]["model_name"],
        image_size=settings["detection_pipeline"]["output_image_size"],
        model_save_path=settings["detection_pipeline"]["pretrained_model_path"],
    )

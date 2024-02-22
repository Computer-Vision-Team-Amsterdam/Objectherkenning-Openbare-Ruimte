import logging
import os
import sys

sys.path.append("../../..")

from aml_interface.azure_logging import setup_azure_logging  # noqa: E402

from objectherkenning_openbare_ruimte.settings.settings import (  # noqa: E402
    ObjectherkenningOpenbareRuimteSettings,
)

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.yml")
)
settings = ObjectherkenningOpenbareRuimteSettings.set_from_yaml(config_path)
aml_experiment_settings = settings["aml_experiment_details"]

# Configure logging
# DO NOT import relative paths before setting up the logger.
# Exception, of course, is settings to set up the logger.
log_settings = ObjectherkenningOpenbareRuimteSettings.set_from_yaml(config_path)[
    "logging"
]
setup_azure_logging(log_settings, __name__)

# from objectherkenning_openbare_ruimte.convert_old_dataset.source.equirectangular_to_cubemap_converter import (  # noqa: E402
#    ConvertEquirectangularImageToCubic,
# )

logger = logging.getLogger("convert_old_dataset")


"""def main(args: argparse.Namespace) -> None:
    input_path = args.input_path
    output_path = args.output_path
    face_width = args.face_width

    total_images = 0
    processed_images = 0
    valid_extensions = {
        ".jpg",
        ".png",
    }

    for img_path in os.listdir(input_path):
        if img_path.endswith(tuple(valid_extensions)):
            total_images += 1

    if os.path.exists(output_path):
        for item in os.listdir(output_path):
            if os.path.isdir(os.path.join(output_path, item)):
                processed_images += 1

    remaining_to_process = total_images - processed_images

    logging.info(f"Total images in input: {total_images}")
    logging.info(f"Processed images in output: {processed_images}")
    logging.info(f"Remaining to process: {remaining_to_process}")

    for img_path in os.listdir(input_path):
        if not img_path.startswith("."):
            if img_path.endswith(".jpg") or img_path.endswith(".png"):

                img = cv2.imread(os.path.join(input_path, img_path))
                if img is None:
                    logging.warning(f"Skipping {img_path} as it is empty or corrupted")
                    continue

                convert_equirectangular_image_to_cubic(
                    input_path, img_path, output_path, face_width
                )

                # Step 3: convert the annotations from yolo to xy coordinates
                process_annotations(input_path, output_path, img_path, face_width)

                if args.visualize_eqr:
                    visualize_annotations_on_equirectangular_image(input_path, img_path)

                if args.visualize_cubemap:
                    img_folder = img_path.split(".")[0]
                    faces = ["top", "bottom", "front", "back", "left", "right"]
                    for face in faces:
                        visualize_annotations_with_corners(
                            output_path, img_folder, face
                        )

                remaining_to_process -= 1
                logging.info(f"Remaining to process: {remaining_to_process}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input image directory.",
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to the output directory."
    )
    parser.add_argument(
        "--face_width",
        type=int,
        default=1024,
        help="Width of each face in the cubemap. Default is 1024.",
    )
    parser.add_argument(
        "--visualize_eqr",
        action="store_true",
        help="Visualize annotations on equirectangular images.",
    )
    parser.add_argument(
        "--visualize_cubemap",
        action="store_true",
        help="Visualize processed annotations on cubic faces.",
    )

    args = parser.parse_args()

    main(args)"""

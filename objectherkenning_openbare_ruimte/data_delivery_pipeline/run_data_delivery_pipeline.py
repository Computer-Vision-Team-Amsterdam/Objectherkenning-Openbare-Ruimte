from objectherkenning_openbare_ruimte.data_delivery_pipeline.components.data_delivery import (
    DataDelivery,
)

if __name__ == "__main__":
    data_delivery_pipeline = DataDelivery(
        images_path="objectherkenning_openbare_ruimte/processed_images",
        detections_path="objectherkenning_openbare_ruimte/processed_images",
        metadata_path="objectherkenning_openbare_ruimte/processed_images",
    )
    data_delivery_pipeline.run_pipeline()

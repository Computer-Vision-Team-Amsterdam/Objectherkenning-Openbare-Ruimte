from datetime import datetime
from typing import Any, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from objectherkenning_openbare_ruimte.databricks_pipelines.common import (
    get_job_process_time,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables import (
    SilverDetectionMetadataManager,
    SilverEnrichedDetectionMetadataManager,
    SilverFrameMetadataManager,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.post_processing_pipeline.data_enrichment_step import (
    Clustering,
    DecosDataHandler,
    StadsdelenHandler,
    VulnerableBridgesHandler,
    utils_scoring,
    utils_visualization,
)


class DataEnrichment:
    def __init__(
        self,
        sparkSession: SparkSession,
        catalog: str,
        schema: str,
        settings: dict[str, Any],
    ):
        self.sparkSession = sparkSession
        self.catalog = catalog
        self.schema = schema
        self.root_source = settings["storage_account_root_path"]
        self.vuln_bridges_relative_path = settings["vuln_bridges_relative_path"]
        self.az_tenant_id = settings["azure_tenant_id"]
        self.db_host = settings["reference_database"]["host"]
        self.db_name = settings["reference_database"]["name"]
        self.device_id = settings["device_id"]
        self.job_process_time = get_job_process_time(
            is_first_pipeline_step=False,
        )
        self.object_classes = settings["job_config"]["object_classes"]["names"]
        self.permit_mapping = settings["job_config"]["object_classes"]["permit_mapping"]
        self.confidence_thresholds = settings["job_config"]["object_classes"][
            "confidence_threshold"
        ]
        self.bbox_size_thresholds = settings["job_config"]["object_classes"][
            "bbox_size_threshold"
        ]
        self.annotate_detection_images = settings["job_config"][
            "annotate_detection_images"
        ]
        self.active_object_classes_for_clustering = settings["job_config"][
            "clustering"
        ]["active_object_classes"]

    def run_data_enrichment_step(self):
        pending_detections = (
            SilverDetectionMetadataManager.load_pending_rows_from_table()
        )
        pending_frames = SilverFrameMetadataManager.load_pending_rows_from_table()

        if pending_detections.count() == 0 or pending_frames.count() == 0:
            print("No pending detections. Exiting.")
        else:
            objects_coordinates_df = self._run_clustering()

            if objects_coordinates_df and (objects_coordinates_df.count() > 0):
                category_counts = sorted(
                    objects_coordinates_df.groupBy("object_class").count().collect()
                )
                for row in category_counts:
                    print(
                        f"Detected '{self.object_classes[row['object_class']]}': {row['count']}"
                    )

                enriched_df = self._get_enriched_df(
                    objects_coordinates_df=objects_coordinates_df
                )

                self._create_map(enriched_df=enriched_df)

                selected_casted_df = enriched_df.select(
                    F.col("a.detection_id").cast("int"),
                    F.col("a.object_class"),
                    F.col("b.gps_lat").alias("object_lat").cast("string"),
                    F.col("b.gps_lon").alias("object_lon").cast("string"),
                    F.col("closest_bridge_distance")
                    .alias("distance_closest_bridge")
                    .cast("float"),
                    F.col("closest_bridge_id").cast("string"),
                    F.col("closest_permit_distance")
                    .alias("distance_closest_permit")
                    .cast("float"),
                    F.col("closest_permit_id"),
                    F.col("closest_permit_lat").cast("float"),
                    F.col("closest_permit_lon").cast("float"),
                    F.col("stadsdeel"),
                    F.col("stadsdeel_code"),
                    F.col("score").cast("float"),
                    F.lit("Pending").alias("status"),
                )

                SilverEnrichedDetectionMetadataManager.insert_data(
                    df=selected_casted_df
                )
            else:
                print("Nothing to do after clustering and filtering. Exiting.")

        SilverFrameMetadataManager.update_status(job_process_time=self.job_process_time)
        SilverDetectionMetadataManager.update_status(
            job_process_time=self.job_process_time
        )

    def _get_enriched_df(self, objects_coordinates_df: DataFrame) -> DataFrame:
        closest_bridges_df = self._get_bridges_df(
            objects_coordinates_df=objects_coordinates_df
        )
        objects_coordinates_with_closest_bridge_df = objects_coordinates_df.join(
            closest_bridges_df, "detection_id"
        )

        closest_permits_df = self._get_decos_df(
            objects_coordinates_df=objects_coordinates_df
        )
        objects_coordinates_with_closest_bridge_permit_df = (
            objects_coordinates_with_closest_bridge_df.join(
                closest_permits_df, "detection_id"
            )
        )

        stadsdelen_df = self._get_stadsdelen_df(
            objects_coordinates_df=objects_coordinates_df
        )
        objects_coordinates_with_closest_bridge_permit_stadsdeel_df = (
            objects_coordinates_with_closest_bridge_permit_df.join(
                stadsdelen_df, "detection_id"
            )
        )

        score_expr = utils_scoring.get_score_expr()
        objects_coordinates_with_closest_bridge_permit_stadsdeel_score_df = (
            objects_coordinates_with_closest_bridge_permit_stadsdeel_df.withColumn(
                "score", score_expr
            )
        )

        joined_metadata_with_details_df = (
            objects_coordinates_with_closest_bridge_permit_stadsdeel_score_df.alias(
                "a"
            ).join(
                self.clustering.joined_metadata.alias("b"),
                on=F.col("a.detection_id") == F.col("b.detection_id"),
            )
        )
        return joined_metadata_with_details_df

    def _run_clustering(self) -> Optional[DataFrame]:
        self.clustering = Clustering(
            spark=self.sparkSession,
            catalog=self.catalog,
            schema=self.schema,
            detections=SilverDetectionMetadataManager.load_pending_rows_from_table(),
            frames=SilverFrameMetadataManager.load_pending_rows_from_table(),
            active_object_classes=self.active_object_classes_for_clustering,
            confidence_thresholds=self.confidence_thresholds,
            bbox_size_thresholds=self.bbox_size_thresholds,
        )
        return self.clustering.get_objects_coordinates_with_detection_id()

    def _get_bridges_df(self, objects_coordinates_df: DataFrame) -> DataFrame:
        bridgesHandler = VulnerableBridgesHandler(
            spark=self.sparkSession,
            root_source=self.root_source,
            vuln_bridges_relative_path=self.vuln_bridges_relative_path,
        )
        bridges_coordinates_geometry = bridgesHandler.get_bridges_coordinates_geometry()
        closest_bridges_df = (
            bridgesHandler.calculate_distances_to_closest_vulnerable_bridges(
                bridges_locations_as_linestrings=bridges_coordinates_geometry,
                objects_coordinates_df=objects_coordinates_df,
                bridges_ids=bridgesHandler.get_bridges_ids(),
                bridges_coordinates=bridgesHandler.get_bridges_coordinates(),
            )
        )
        return closest_bridges_df

    def _get_decos_df(self, objects_coordinates_df: DataFrame) -> DataFrame:
        decosDataHandler = DecosDataHandler(
            spark=self.sparkSession,
            az_tenant_id=self.az_tenant_id,
            db_host=self.db_host,
            db_name=self.db_name,
            db_port=5432,
            object_classes=self.object_classes,
            permit_mapping=self.permit_mapping,
        )
        decosDataHandler.query_and_process_object_permits(
            date_to_query=datetime.today().strftime("%Y-%m-%d")
        )
        closest_permits_df = decosDataHandler.calculate_distances_to_closest_permits(
            objects_coordinates_df=objects_coordinates_df,
        )
        return closest_permits_df

    def _get_stadsdelen_df(self, objects_coordinates_df: DataFrame) -> DataFrame:
        stadsdelenHandler = StadsdelenHandler(spark_session=self.sparkSession)
        stadsdelen_df = stadsdelenHandler.lookup_stadsdeel_for_detections(
            objects_coordinates_df=objects_coordinates_df
        )
        return stadsdelen_df

    def _create_map(self, enriched_df: DataFrame) -> None:
        utils_visualization.generate_map(
            dataframe=enriched_df,
            annotate_detection_images=self.annotate_detection_images,
            name=f"{self.job_process_time}-map",
            path=f"/Volumes/{self.catalog}/default/landingzone/Luna/visualizations/{datetime.today().strftime('%Y-%m-%d')}/",
            catalog=self.catalog,
            device_id=self.device_id,
        )

from datetime import datetime
from typing import Any, Dict, Optional

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
    PrivateTerrainHandler,
    StadsdelenHandler,
    VulnerableBridgesHandler,
    utils_scoring,
    utils_visualization,
)


class DataEnrichment:
    id_column = "detection_id"

    def __init__(
        self,
        spark_session: SparkSession,
        catalog: str,
        schema: str,
        settings: dict[str, Any],
    ):
        self.spark_session = spark_session
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

        object_class_settings: Dict[str, Dict[int, Any]] = settings["job_config"][
            "object_classes"
        ]
        self.object_classes = object_class_settings["names"]
        self.permit_mapping = object_class_settings["permit_mapping"]
        self.confidence_thresholds = object_class_settings["confidence_threshold"]
        self.bbox_size_thresholds = object_class_settings["bbox_size_threshold"]
        self.cluster_distances = object_class_settings["cluster_distances"]
        self.active_object_classes_for_clustering = list(self.cluster_distances.keys())

        job_settings = settings["job_config"]
        self.annotate_detection_images = job_settings["annotate_detection_images"]
        self.exclude_private_terrain = job_settings[
            "exclude_private_terrain_detections"
        ]
        self.public_terrain_detection_buffer = job_settings[
            "private_terrain_detection_buffer"
        ]

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
                    F.col("a.detection_id"),
                    F.col("a.object_class"),
                    F.col("b.gps_lat").alias("object_lat"),
                    F.col("b.gps_lon").alias("object_lon"),
                    F.col("closest_bridge_distance")
                    .alias("distance_closest_bridge")
                    .cast("float"),
                    F.col("closest_bridge_id").cast("string"),
                    F.col("closest_permit_distance")
                    .alias("distance_closest_permit")
                    .cast("float"),
                    F.col("closest_permit_id").cast("string"),
                    F.col("closest_permit_lat").cast("double"),
                    F.col("closest_permit_lon").cast("double"),
                    F.col("stadsdeel"),
                    F.col("stadsdeel_code"),
                    F.col("score").cast("float"),
                    F.col("private_terrain").cast("boolean"),
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
        objects_coordinates_enriched_df = objects_coordinates_df.join(
            closest_bridges_df, self.id_column
        )

        closest_permits_df = self._get_decos_df(
            objects_coordinates_df=objects_coordinates_df
        )
        objects_coordinates_enriched_df = objects_coordinates_enriched_df.join(
            closest_permits_df, self.id_column
        )

        stadsdelen_df = self._get_stadsdelen_df(
            objects_coordinates_df=objects_coordinates_df
        )
        objects_coordinates_enriched_df = objects_coordinates_enriched_df.join(
            stadsdelen_df, self.id_column
        )

        private_terrain_df = self._get_private_terrain_df(
            objects_coordinates_df=objects_coordinates_df
        )
        objects_coordinates_enriched_df = objects_coordinates_enriched_df.join(
            private_terrain_df, self.id_column
        )

        score_expr = utils_scoring.get_score_expr()
        objects_coordinates_enriched_df = objects_coordinates_enriched_df.withColumn(
            "score", score_expr
        )

        joined_metadata_with_details_df = objects_coordinates_enriched_df.alias(
            "a"
        ).join(
            self.clustering.joined_metadata.alias("b"),
            on=F.col(f"a.{self.id_column}") == F.col(f"b.{self.id_column}"),
        )
        return joined_metadata_with_details_df

    def _run_clustering(self) -> Optional[DataFrame]:
        self.clustering = Clustering(
            spark_session=self.spark_session,
            catalog=self.catalog,
            schema=self.schema,
            detections=SilverDetectionMetadataManager.load_pending_rows_from_table(),
            frames=SilverFrameMetadataManager.load_pending_rows_from_table(),
            active_object_classes=self.active_object_classes_for_clustering,
            confidence_thresholds=self.confidence_thresholds,
            bbox_size_thresholds=self.bbox_size_thresholds,
            cluster_distances=self.cluster_distances,
        )
        return self.clustering.get_objects_coordinates_with_detection_id()

    def _get_bridges_df(self, objects_coordinates_df: DataFrame) -> DataFrame:
        bridges_handler = VulnerableBridgesHandler(
            spark_session=self.spark_session,
            root_source=self.root_source,
            vuln_bridges_relative_path=self.vuln_bridges_relative_path,
        )
        bridges_coordinates_geometry = (
            bridges_handler.get_bridges_coordinates_geometry()
        )
        closest_bridges_df = (
            bridges_handler.calculate_distances_to_closest_vulnerable_bridges(
                bridges_locations_as_linestrings=bridges_coordinates_geometry,
                objects_coordinates_df=objects_coordinates_df,
                bridges_ids=bridges_handler.get_bridges_ids(),
                bridges_coordinates=bridges_handler.get_bridges_coordinates(),
            )
        )
        return closest_bridges_df

    def _get_decos_df(self, objects_coordinates_df: DataFrame) -> DataFrame:
        decos_data_handler = DecosDataHandler(
            spark_session=self.spark_session,
            az_tenant_id=self.az_tenant_id,
            db_host=self.db_host,
            db_name=self.db_name,
            object_classes=self.object_classes,
            permit_mapping=self.permit_mapping,
        )
        decos_data_handler.query_and_process_object_permits(
            date_to_query=datetime.today().strftime("%Y-%m-%d")
        )
        closest_permits_df = decos_data_handler.calculate_distances_to_closest_permits(
            objects_coordinates_df=objects_coordinates_df,
        )
        return closest_permits_df

    def _get_stadsdelen_df(self, objects_coordinates_df: DataFrame) -> DataFrame:
        stadsdelen_handler = StadsdelenHandler(spark_session=self.spark_session)
        stadsdelen_df = stadsdelen_handler.lookup_stadsdeel_for_detections(
            objects_coordinates_df=objects_coordinates_df,
            id_column=self.id_column,
        )
        return stadsdelen_df

    def _get_private_terrain_df(self, objects_coordinates_df: DataFrame) -> DataFrame:
        private_terrain_handler = PrivateTerrainHandler(
            spark_session=self.spark_session,
            az_tenant_id=self.az_tenant_id,
            db_host=self.db_host,
            db_name=self.db_name,
            detection_buffer=self.public_terrain_detection_buffer,
        )
        private_terrain_df = (
            private_terrain_handler.lookup_private_terrain_for_detections(
                objects_coordinates_df=objects_coordinates_df,
                id_column=self.id_column,
            )
        )
        return private_terrain_df

    def _create_map(self, enriched_df: DataFrame) -> None:
        map_file_name = f"{self.job_process_time.strftime('%Y-%m-%d %Hh%Mm%Ss')}-map"
        map_file_path = f"/Volumes/{self.catalog}/default/landingzone/{self.device_id}/visualizations/{datetime.today().strftime('%Y-%m-%d')}/"

        utils_visualization.generate_map(
            dataframe=enriched_df,
            file_name=map_file_name,
            file_path=map_file_path,
            catalog=self.catalog,
            device_id=self.device_id,
            annotate_detection_images=self.annotate_detection_images,
            exclude_private_terrain=self.exclude_private_terrain,
        )

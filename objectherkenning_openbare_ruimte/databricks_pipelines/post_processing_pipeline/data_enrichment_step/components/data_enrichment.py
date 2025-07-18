import datetime
from functools import reduce
from typing import Any, Dict, List, Optional, Tuple

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
    date_column = "gps_timestamp"
    date_format = "yyyy-MM-dd"

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

        job_settings: Dict[str, Any] = settings["job_config"]
        self.annotate_detection_images = job_settings["annotate_detection_images"]
        self.exclude_private_terrain = job_settings[
            "exclude_private_terrain_detections"
        ]
        self.public_terrain_detection_buffer = job_settings[
            "private_terrain_detection_buffer"
        ]
        self.min_score = job_settings["min_score"]
        self.detection_date: datetime.date = job_settings.get("detection_date", None)

    def run_data_enrichment_step(self):
        """
        Load pending detections, group them by date, and then run data enrichment steps:
        - Clustering of detections
        - Add distance to nearest vulnerable bridge
        - Find closest matching permit
        - Add name of Stadsdeel within which the detection is located
        - Check whether the detection is located on private terrain

        A map visualization will be created for each date separately.
        """
        pending_detections = (
            SilverDetectionMetadataManager.load_pending_rows_from_table()
        )
        pending_frames = SilverFrameMetadataManager.load_pending_rows_from_table()

        if pending_detections.count() == 0 or pending_frames.count() == 0:
            print("\n=== No pending detections. Exiting. ===")
        else:
            pending_dates = self._get_pending_dates(pending_frames)

            if len(pending_dates) == 0:
                print(
                    f"\n=== No pending detections that match date {self.detection_date}. Exiting. ==="
                )
            else:
                self._setup_handlers()

                enriched_dfs = []

                # Loop over dates and enrich corresponding detections
                for date in pending_dates:
                    print(f"\n=== Processing data for date: {date} ===")
                    enriched_df = self._process_pending_detections_for_date(
                        pending_frames, pending_detections, date
                    )
                    if enriched_df is not None:
                        enriched_dfs.append(enriched_df)

                if len(enriched_dfs) > 0:
                    merged_enriched_df = reduce(DataFrame.unionAll, enriched_dfs)

                    if merged_enriched_df.count() > 0:
                        selected_casted_df = merged_enriched_df.select(
                            F.col("a.detection_id"),
                            F.col("detection_date"),
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
                    print("No new enriched metadata to add to table.")

        # If a specific date was set, we need to keep track of the pending
        # detection ids that were actually processed, and leave the rest on
        # "pending"
        filtered_frame_ids, filtered_detection_ids = None, None
        if self.detection_date is not None:
            filtered_frame_ids, filtered_detection_ids = self._get_ids_for_date(
                pending_frames,
                pending_detections,
                self.detection_date.strftime("%Y-%m-%d"),
            )

        SilverFrameMetadataManager.update_status(
            job_process_time=self.job_process_time,
            only_ids=filtered_frame_ids,
        )
        SilverDetectionMetadataManager.update_status(
            job_process_time=self.job_process_time, only_ids=filtered_detection_ids
        )

    def _process_pending_detections_for_date(
        self, pending_frames: DataFrame, pending_detections: DataFrame, date: str
    ) -> Optional[DataFrame]:
        """
        Process pending detections corresponding to a specific date (format: "yyyy-mm-dd").
        """
        frames_for_date, detections_for_date = self._filter_by_date(
            pending_frames, pending_detections, date
        )
        objects_coordinates_df = self._run_clustering(
            pending_detections=detections_for_date,
            pending_frames=frames_for_date,
        )

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

            if enriched_df is not None:
                enriched_df = enriched_df.withColumn(
                    "detection_date", F.to_date(F.lit(date), self.date_format)
                )
                self._create_map(enriched_df=enriched_df, date=date)
            else:
                print("No detections left after enrichment.")

            return enriched_df
        else:
            print("Nothing to do after clustering and filtering.")
            return None

    def _get_pending_dates(self, pending_frames: DataFrame) -> List[str]:
        """
        Returns a list of unique pending dates (format: "yyyy-mm-dd"). If a
        detection date was specified as a parameter, the chosen date is
        returned, only if it is included in the pending dates.
        """
        dates = (
            pending_frames.select(F.date_format(self.date_column, self.date_format))
            .distinct()
            .rdd.flatMap(lambda x: x)
            .collect()
        )
        if self.detection_date is not None:
            _detection_date = self.detection_date.strftime("%Y-%m-%d")
            dates = list(set([_detection_date]).intersection(dates))
        return sorted(dates)

    def _filter_by_date(
        self, pending_frames: DataFrame, pending_detections: DataFrame, date: str
    ) -> Tuple[DataFrame, DataFrame]:
        """Filter pending frames and detections by date."""
        filtered_frames = pending_frames.filter(
            F.date_format(self.date_column, self.date_format) == date
        )
        filtered_detections = pending_detections.join(
            other=filtered_frames,
            on="frame_id",
            how="left_semi",
        )
        return filtered_frames, filtered_detections

    def _get_ids_for_date(
        self, pending_frames: DataFrame, pending_detections: DataFrame, date: str
    ) -> Tuple[List[int], List[int]]:
        """
        Returns two lists [frame_id] and [detection_id] of pending frames and
        detections corresponding to a specified date.
        """
        filtered_frames, filtered_detections = self._filter_by_date(
            pending_frames, pending_detections, date
        )
        filtered_frame_ids = (
            filtered_frames.select("frame_id").rdd.flatMap(lambda x: x).collect()
        )
        filtered_detection_ids = (
            filtered_detections.select("detection_id")
            .rdd.flatMap(lambda x: x)
            .collect()
        )
        return filtered_frame_ids, filtered_detection_ids

    def _setup_handlers(self) -> None:
        """
        Setup data handlers. We do this only once for each run for efficiency,
        since each handler collects and pre-processes potentially a lot a data.
        """
        self.bridges_handler = VulnerableBridgesHandler(
            spark_session=self.spark_session,
            root_source=self.root_source,
            vuln_bridges_relative_path=self.vuln_bridges_relative_path,
        )

        self.decos_data_handler = DecosDataHandler(
            spark_session=self.spark_session,
            az_tenant_id=self.az_tenant_id,
            db_host=self.db_host,
            db_name=self.db_name,
            object_classes=self.object_classes,
            permit_mapping=self.permit_mapping,
        )
        self.decos_data_handler.query_and_process_object_permits(
            date_to_query=datetime.datetime.today().strftime("%Y-%m-%d")
        )

        self.stadsdelen_handler = StadsdelenHandler(spark_session=self.spark_session)

        self.private_terrain_handler = PrivateTerrainHandler(
            spark_session=self.spark_session,
            az_tenant_id=self.az_tenant_id,
            db_host=self.db_host,
            db_name=self.db_name,
            detection_buffer=self.public_terrain_detection_buffer,
        )

    def _get_enriched_df(
        self, objects_coordinates_df: DataFrame
    ) -> Optional[DataFrame]:
        """
        Enrich the detections with bridges, permits, stadsdelen, and private
        terrain information.
        """
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

        if objects_coordinates_enriched_df.count() > 0:
            score_expr = utils_scoring.get_score_expr()
            objects_coordinates_enriched_df = (
                objects_coordinates_enriched_df.withColumn("score", score_expr)
            )

            joined_metadata_with_details_df = objects_coordinates_enriched_df.alias(
                "a"
            ).join(
                self.clustering.joined_metadata.alias("b"),
                on=F.col(f"a.{self.id_column}") == F.col(f"b.{self.id_column}"),
            )
            return joined_metadata_with_details_df
        else:
            return None

    def _run_clustering(
        self, pending_detections: DataFrame, pending_frames: DataFrame
    ) -> Optional[DataFrame]:
        self.clustering = Clustering(
            spark_session=self.spark_session,
            catalog=self.catalog,
            schema=self.schema,
            detections=pending_detections,
            frames=pending_frames,
            active_object_classes=self.active_object_classes_for_clustering,
            confidence_thresholds=self.confidence_thresholds,
            bbox_size_thresholds=self.bbox_size_thresholds,
            cluster_distances=self.cluster_distances,
        )
        return self.clustering.get_objects_coordinates_with_detection_id()

    def _get_bridges_df(self, objects_coordinates_df: DataFrame) -> DataFrame:
        closest_bridges_df = self.bridges_handler.calculate_distances_to_closest_vulnerable_bridges(
            bridges_locations_as_linestrings=self.bridges_handler.get_bridges_coordinates_geometry(),
            objects_coordinates_df=objects_coordinates_df,
            bridges_ids=self.bridges_handler.get_bridges_ids(),
            bridges_coordinates=self.bridges_handler.get_bridges_coordinates(),
        )
        return closest_bridges_df

    def _get_decos_df(self, objects_coordinates_df: DataFrame) -> DataFrame:
        closest_permits_df = (
            self.decos_data_handler.calculate_distances_to_closest_permits(
                objects_coordinates_df=objects_coordinates_df,
            )
        )
        return closest_permits_df

    def _get_stadsdelen_df(self, objects_coordinates_df: DataFrame) -> DataFrame:
        stadsdelen_df = self.stadsdelen_handler.lookup_stadsdeel_for_detections(
            objects_coordinates_df=objects_coordinates_df,
            id_column=self.id_column,
        )
        return stadsdelen_df

    def _get_private_terrain_df(self, objects_coordinates_df: DataFrame) -> DataFrame:
        private_terrain_df = (
            self.private_terrain_handler.lookup_private_terrain_for_detections(
                objects_coordinates_df=objects_coordinates_df,
                id_column=self.id_column,
            )
        )
        return private_terrain_df

    def _create_map(self, enriched_df: DataFrame, date: str) -> None:
        map_file_name = f"map-{date}-created-at-{self.job_process_time.strftime('%Y-%m-%d %Hh%Mm%Ss')}"
        map_file_path = f"/Volumes/{self.catalog}/default/landingzone/{self.device_id}/visualizations/{datetime.datetime.today().strftime('%Y-%m-%d')}/"

        map_df = enriched_df.filter(F.col("score") >= self.min_score)

        utils_visualization.generate_map(
            dataframe=map_df,
            file_name=map_file_name,
            file_path=map_file_path,
            catalog=self.catalog,
            device_id=self.device_id,
            annotate_detection_images=self.annotate_detection_images,
            exclude_private_terrain=self.exclude_private_terrain,
        )

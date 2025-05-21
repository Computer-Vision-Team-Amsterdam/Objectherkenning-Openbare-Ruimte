import os
import tempfile
from datetime import datetime
from typing import List

from pyspark.sql import DataFrame, SparkSession

from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables import (
    BronzeFrameMetadataManager,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.post_processing_pipeline.ingest_metadata_step.components.json_frame_detection_adapter import (
    JsonFrameDetectionAdapter,
)


class DataLoader:

    def __init__(
        self,
        spark_session: SparkSession,
        catalog: str,
        schema: str,
        root_source: str,
        device_id: str,
        ckpt_frames_relative_path: str,
        ckpt_detections_relative_path: str,
        job_process_time: datetime,
    ):
        self.spark_session = spark_session
        self.catalog = catalog
        self.schema = schema
        self.root_source = root_source
        self.device_id = device_id
        self.checkpoint_frames = (
            f"{self.root_source}/{self.device_id}/{ckpt_frames_relative_path}"
        )
        self.checkpoint_detections = (
            f"{self.root_source}/{self.device_id}/{ckpt_detections_relative_path}"
        )
        self.frame_metadata_table = (
            f"{self.catalog}.{self.schema}.bronze_frame_metadata"
        )
        self.detection_metadata_table = (
            f"{self.catalog}.{self.schema}.bronze_detection_metadata"
        )
        self.temp_files: List[str] = []
        self.job_process_time = job_process_time

    def _get_schema_path(self, table_name: str) -> str:
        """
        Retrieves the schema of the specified table and saves it to a temporary file.

        Parameters:
            table_name (str): The name of the table.

        Returns:
            str: The path to the temporary file containing the schema JSON.
        """
        # Retrieve the schema of the specified table
        existing_table_schema = self.spark_session.table(table_name).schema
        schema_json = existing_table_schema.json()

        # Save the JSON schema to a temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
            temp_file.write(schema_json)
            path_table_schema = temp_file.name

        self.temp_files.append(path_table_schema)
        return path_table_schema

    def ingest_json_metadata(self):
        """
        Ingest metadata from JSON files, convert to the old frame_metadata
        and detection_metadata format, and store in bronze tables.
        """

        json_source = f"{self.root_source}/{self.device_id}/detection_metadata"

        # 1) Generate two schema-tracking locations
        frame_schema_loc = self._get_schema_path(self.frame_metadata_table)
        detection_schema_loc = self._get_schema_path(self.detection_metadata_table)

        # 2) Read JSON data twice, each with its own schemaLocation
        adapter = JsonFrameDetectionAdapter(
            spark=self.spark_session,
            json_source=json_source,
            frame_schema_loc=frame_schema_loc,
            detection_schema_loc=detection_schema_loc,
        )

        # 3) Transform data
        frames_df = adapter.to_frame_df()
        detections_df = adapter.to_det_df()

        print("Loaded JSON metadata.")

        # 4) Store new data
        self._store_new_data(
            frames_df,
            checkpoint_path=self.checkpoint_frames,
            target=self.frame_metadata_table,
        )

        detections_df_with_frame_id = self._match_frame_ids_to_detections(detections_df)
        self._store_new_data(
            detections_df_with_frame_id,
            checkpoint_path=self.checkpoint_detections,
            target=self.detection_metadata_table,
        )

    def _match_frame_ids_to_detections(self, detections_df: DataFrame) -> DataFrame:
        pending_frames_df = BronzeFrameMetadataManager.load_pending_rows_from_table()

        detections_df_with_frame_id = detections_df.drop("frame_id").join(
            pending_frames_df.select("image_name", "fame_id"),
            on="image_name",
            how="left",
        )

        return detections_df_with_frame_id

    def _store_new_data(self, df, checkpoint_path: str, target: str):
        # availableNow = process all files that have been added before the time when this query ran. Used with batch processing
        stream_query = (
            df.writeStream.option("checkpointLocation", checkpoint_path)
            .trigger(availableNow=True)
            .toTable(target)
        )

        query_progress = stream_query.awaitTermination(60)

        # Get number of rows processed
        if query_progress:
            rows_processed = stream_query.lastProgress["numInputRows"]
            print(f"Stored {rows_processed} new rows into {target}.")
        else:
            print(
                f"Query did not terminate properly for checkpointLocation {checkpoint_path} and target table {target}."
            )

    def cleanup_temp_files(self):
        """
        Deletes all temporary files created during the processing.
        """
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"Deleted temporary file: {temp_file}")

from pyspark.sql.functions import col



class MetadataHealthChecker:
    def __init__(self):
        self.load_bronze_metadata()
        self.process_and_save_metadata()
        self.update_bronze_status()


    def load_bronze_metadata(self):
        query = "SELECT * FROM dpcv_dev.oor.bronze_frame_metadata WHERE status='Pending'"
        self.bronze_frame_metadata = spark.sql(query)

    def process_and_save_metadata(self):
        valid_metadata = self.bronze_frame_metadata.filter((col('gps_lat').isNotNull()) & (col('gps_lat') != 0) &
                                    (col('gps_lon').isNotNull()) & (col('gps_lon') != 0))
        
        invalid_metadata = self.bronze_frame_metadata.filter((col('gps_lat').isNull()) | (col('gps_lat') == 0) |
                                        (col('gps_lon').isNull()) | (col('gps_lon') == 0))
        
        
        valid_metadata.write.mode('append').saveAsTable('dpcv_dev.oor.silver_frame_metadata')
        
        invalid_metadata.write.mode('append').saveAsTable('dpcv_dev.oor.silver_frame_metadata_quarantine')


    def update_bronze_status(self):
        
        # Update the status to "processed"
        processed_metadata = self.bronze_frame_metadata.withColumn('status', col('status').substr(0, 0).lit('Processed'))
        
    def update_bronze_status(self):
        # Update the status of the rows where status is 'Pending'
        update_query = """
        INSERT OVERWRITE TABLE dpcv_dev.oor.bronze_frame_metadata
        SELECT
            *
            CASE
                WHEN status = 'Pending' THEN 'Processed'
                ELSE status
            END as status,
            gps_lat,
            gps_lon
        FROM dpcv_dev.oor.bronze_frame_metadata
        """
        
        # Execute the update query
        spark.sql(update_query)

if __name__ == "__main__":
    metadataHealthChecker = MetadataHealthChecker()
    
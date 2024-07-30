# this fixes the caching issues, reimports all modules
dbutils.library.restartPython()

from datetime import datetime
from helpers.utils_signalen import SignalHandler
from helpers.databricks_workspace import get_catalog_name
from helpers.databricks_workspace import get_databricks_environment, get_job_process_time # noqa: E402
from pyspark.sql import SparkSession, Row
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField

def run_submit_to_signalen_step(sparkSession, catalog, schema, client_id, client_secret_name, access_token_url, base_url):
   signalHandler = SignalHandler(sparkSession, catalog, schema, client_id, client_secret_name, access_token_url, base_url)

    gold_signal_notifications = sparkSession.table(f"{signalHandler.catalog_name}.oor.gold_signal_notifications")
        
   top_scores_df= signalHandler.get_top_pending_records(table_name="silver_objects_per_day", limit=10)

   if top_scores_df.count() == 0:
        print("04: No data found for creating notifications. Stopping execution.")
        return
        
   print(f"04: Loaded {top_scores_df.count()} rows with top 10 scores from {signalHandler.catalog_name}.oor.silver_objects_per_day.")

   date_of_notification = datetime.today().strftime('%Y-%m-%d')
   top_scores_df_with_date = top_scores_df.withColumn("notification_date", F.to_date(F.lit(date_of_notification)))

   successful_notifications = []
   unsuccessful_notifications = []

   # Convert to a list of row objects that are iterable
   for entry in top_scores_df_with_date.collect():

      LAT = float(entry["object_lat"])
      LON = float(entry["object_lon"])
      detection_id = entry["detection_id"]

      image_upload_path = signalHandler.get_image_upload_path(detection_id=detection_id)
      entry_dict = entry.asDict()
      entry_dict.pop('processed_at', None)
      entry_dict.pop('id', None)
      try:
         # Check if image exists
         dbutils.fs.head(image_upload_path)
         notification_json = SignalHandler.fill_incident_details(incident_date=date_of_notification, lon=LON, lat=LAT,)
         id = signalHandler.post_signal_with_image_attachment(json_content=notification_json, filename=image_upload_path)
         print(f"Created signal {id} with image on {date_of_notification} with lat {LAT} and lon {LON}.\n\n" )
         entry_dict['signal_id'] = id
         updated_entry = Row(**entry_dict)         
         successful_notifications.append(updated_entry)
      except Exception as e:

         entry_dict.pop('notification_date', None)  #
         updated_failed_entry = Row(**entry_dict)
         if 'java.io.FileNotFoundException' in str(e):
            print(f"Image not found: {image_upload_path}. Skip creating notification...\n\n")
            unsuccessful_notifications.append(updated_failed_entry)

         else:
            print(f"An error occurred: {e}\n\n")
            unsuccessful_notifications.append(updated_failed_entry)
   
   if successful_notifications:
      # Remove 'processed_at' field from schema
      modified_schema = StructType([field for field in gold_signal_notifications.schema if field.name not in {'id', 'processed_at'}])
      successful_df = spark.createDataFrame(successful_notifications, schema=modified_schema) 
      successful_df.write.mode('append').saveAsTable(f'{signalHandler.catalog_name}.oor.gold_signal_notifications')
      print(f"04: Appended {len(successful_notifications)} rows to gold_signal_notifications.")
   else:
      print("Appended 0 rows to gold_signal_notifications.")
   
   if unsuccessful_notifications:
      modified_schema = StructType([field for field in top_scores_df.schema if field.name not in { 'id', 'processed_at'}])
      unsuccessful_df = spark.createDataFrame(unsuccessful_notifications, schema=modified_schema)
      print(f"{unsuccessful_df.count()} unsuccessful notifications.")
      unsuccessful_df.write.mode('append').saveAsTable(f'{signalHandler.catalog_name}.oor.silver_objects_per_day_quarantine')
      print(f"04: Appended {len(unsuccessful_notifications)} rows to silver_objects_per_day_quarantine.")
   
   signalHandler.update_status(table_name="silver_objects_per_day") 


def again():

    # Initialize SignalHandler
    signalHandler = SignalHandler(sparkSession, catalog, schema, client_id, client_secret_name, access_token_url, base_url)
    
    # Get top pending records
    top_scores_df = signalHandler.get_top_pending_records(table_name="silver_objects_per_day", limit=10)
    
    # Check if there are records to process
    if top_scores_df.count() == 0:
        print("04: No data found for creating notifications. Stopping execution.")
        return
    
    print(f"04: Loaded {top_scores_df.count()} rows with top 10 scores from {signalHandler.catalog_name}.oor.silver_objects_per_day.")
    
    # Process notifications
    successful_notifications, unsuccessful_notifications = signalHandler.process_notifications(top_scores_df)
    
    # Save notifications
    signalHandler.save_notifications(successful_notifications, unsuccessful_notifications)
    
    # Update status
    signalHandler.update_status(table_name="silver_objects_per_day")

if __name__ == "__main__":
   sparkSession = SparkSession.builder.appName("SignalHandler").getOrCreate()
   databricks_environment = get_databricks_environment(sparkSession)
    settings = load_settings("../../config.yml")["databricks_pipelines"][f"{databricks_environment}"]
   run_submit_to_signalen_step(
      sparkSession=sparkSession, 
      catalog=settings["catalog"],
      schema=settings["schema"],
      client_id=settings["signalen"]["client_id"],
      client_secret_name=settings["signalen"]["client_secret_name"],
      access_token_url=settings["signalen"]["access_token_url"],
      base_url=settings["signalen"]["base_url"])

   
# this fixes the caching issues, reimports all modules
dbutils.library.restartPython()

from datetime import datetime
from helpers.utils_signalen import SignalHandler
from helpers.databricks_workspace import get_catalog_name
from helpers.databricks_workspace import get_databricks_environment, get_job_process_time # noqa: E402
from objectherkenning_openbare_ruimte.settings.databricks_jobs_settings import load_settings

from pyspark.sql import SparkSession, Row
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField

def run_submit_to_signalen_step(sparkSession, catalog, schema, client_id, client_secret_name, access_token_url, base_url):
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

   
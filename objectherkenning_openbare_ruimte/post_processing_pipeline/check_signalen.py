# this fixes the caching issues, reimports all modules
dbutils.library.restartPython()

from helpers.utils_signalen import SignalHandler
from pyspark.sql import SparkSession

sparkSession = SparkSession.builder.appName("Testing").getOrCreate()

signalHandler = SignalHandler(sparkSession)
id = signalHandler.get_signal(sig_id="2230098")

print(id)

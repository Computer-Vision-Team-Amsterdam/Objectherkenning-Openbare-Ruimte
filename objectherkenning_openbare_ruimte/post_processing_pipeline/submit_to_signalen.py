from utils_signalen import SignalHandler

LAT = 52.368920318405074  
LON = 4.9031888157681935 
date_of_notification = "2024-05-24"
image_to_upload = "/Volumes/dpcv_dev/default/landingzone/test-diana/images/D14M03Y2024/1-0-D14M03Y2024-H13M01S13_0080.jpg"


notification_json = SignalHandler.fill_incident_details(incident_date=date_of_notification,
                                                       lon=LON,
                                                       lat=LAT,
                                                       )

signalHandler = SignalHandler()
signalHandler.post_signal_with_image_attachment(json_content=notification_json, filename=image_to_upload)
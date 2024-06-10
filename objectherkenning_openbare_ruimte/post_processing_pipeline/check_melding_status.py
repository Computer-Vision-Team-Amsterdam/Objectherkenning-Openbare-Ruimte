from utils_signalen import SignalHandler

signalHandler = SignalHandler()

notification = signalHandler.get_signal(sig_id="16728")

print(notification["status"])
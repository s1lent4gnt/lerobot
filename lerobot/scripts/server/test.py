import logging

logging.basicConfig(filename="/var/log/test_python.log", level=logging.DEBUG)
logging.info("Test log entry from Python")

print("Log written to /var/log/test_python.log")
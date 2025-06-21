import logging

logging.basicConfig(filename="thesis.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
logger = logging.getLogger("thesis")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    handler = logging.StreamHandler()
    logger.addHandler(handler)
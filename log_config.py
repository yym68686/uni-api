import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("uni-api")

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.CRITICAL)
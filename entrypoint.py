"""
Entrypoint for the container.

Try import key components and run an infinite loop with heartbeat.
"""

import logging
import time
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("rich")

def main():
    logger.info("Greetings from the entrypoint!")

    logger.info("Importing vllm...")
    import vllm
    logger.info(vllm.__version__)

    logger.info("Importing torch...")
    import torch
    logger.info(torch.__version__)
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    while True:
        logger.info("Heartbeat")
        time.sleep(60)
    
    logger.info("Exiting...")


if __name__ == "__main__":
    main()

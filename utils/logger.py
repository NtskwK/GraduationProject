from loguru import logger
import os
import time

class CliLogger:
    def __init__(self, log_dir="logs", log_level="DEBUG"):
        # 移除默认的日志处理器
        logger.remove()

        # 添加控制台日志处理器
        logger.add(
            sink=lambda msg: print(msg, end=""),
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=log_level,
        )

        self.logger = logger

    def get_logger(self):
        return self.logger
import logging
import colorlog

# 로그 포맷 정의
log_formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red'
    }
)

# 콘솔 핸들러 설정
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

# 로거 설정
logger = logging.getLogger("my_logger")
logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)
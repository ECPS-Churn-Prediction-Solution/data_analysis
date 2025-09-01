# -*- coding: utf-8 -*-
import logging, os, sys

LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
FORMAT = os.getenv("LOG_FORMAT", "plain")  # "json"도 가능하도록 확장 여지

class _JsonFormatter(logging.Formatter):
    def format(self, record):
        import json, time
        base = {
            "level": record.levelname,
            "time":  int(record.created * 1000),
            "name":  record.name,
            "msg":   record.getMessage(),
        }
        if record.exc_info:
            base["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(base, ensure_ascii=False)

def get_logger(name: str = "ecps"):
    logger = logging.getLogger(name)
    if logger.handlers:  # 중복 핸들러 방지
        return logger
    logger.setLevel(LEVEL)
    h = logging.StreamHandler(sys.stdout)
    if FORMAT == "json":
        h.setFormatter(_JsonFormatter())
    else:
        h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s - %(message)s"))
    logger.addHandler(h)
    return logger

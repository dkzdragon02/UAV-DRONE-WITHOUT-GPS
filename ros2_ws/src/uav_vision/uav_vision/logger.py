import logging
import logging.handlers
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import os

class StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
        
        return json.dumps(log_data, default=str)
    
    def formatException(self, exc_info):
        import traceback
        return traceback.format_exception(*exc_info)

class RotatingFileHandler(logging.handlers.RotatingFileHandler):
    def __init__(
        self,
        filename: str,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        encoding: str = 'utf-8'
    ):
        log_dir = Path(filename).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        super().__init__(
            filename,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding=encoding
        )

class LoggerManager:
    def __init__(
        self,
        log_dir: str = "logs",
        log_level: str = "INFO",
        use_json: bool = True,
        console_output: bool = True
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.use_json = use_json
        self.console_output = console_output
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        
        self._configured_loggers: Dict[str, logging.Logger] = {}
    
    def get_logger(
        self,
        name: str,
        log_file: Optional[str] = None
    ) -> logging.Logger:

        if name in self._configured_loggers:
            return self._configured_loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(self.log_level)
        logger.handlers.clear()             # Remove default handlers
        
        if log_file is None:
            log_file = f"{name}.log"
        
        log_path = self.log_dir / log_file
        
        file_handler = RotatingFileHandler(
            str(log_path),
            max_bytes=10 * 1024 * 1024,     # 10MB
            backup_count=5
        )
        
        if self.use_json:
            file_handler.setFormatter(StructuredFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            )
        
        logger.addHandler(file_handler)

        if self.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            if self.use_json:
                console_handler.setFormatter(StructuredFormatter())
            else:
                console_handler.setFormatter(
                    logging.Formatter(
                        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    )
                )
            logger.addHandler(console_handler)
        
        self._configured_loggers[name] = logger
        return logger
    
    def set_level(self, level: str):
        log_level = getattr(logging, level.upper(), logging.INFO)
        for logger in self._configured_loggers.values():
            logger.setLevel(log_level)

def get_logger(
    name: str,
    log_dir: str = "logs",
    use_json: bool = True
) -> logging.Logger:

    manager = LoggerManager(log_dir=log_dir, use_json=use_json)
    return manager.get_logger(name)


class PerformanceLogger:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_metric(
        self,
        metric_name: str,
        value: float,
        unit: str = "",
        tags: Optional[Dict[str, str]] = None
    ):
        extra_fields = {
            'metric_name': metric_name,
            'metric_value': value,
            'metric_unit': unit,
            'metric_type': 'performance'
        }
        
        if tags:
            extra_fields['tags'] = tags
        
        record = logging.LogRecord(
            name=self.logger.name,
            level=logging.INFO,
            pathname='',
            lineno=0,
            msg=f"Metric: {metric_name} = {value} {unit}",
            args=(),
            exc_info=None
        )
        record.extra_fields = extra_fields
        self.logger.handle(record)
    
    def log_timing(
        self,
        operation: str,
        duration: float,
        success: bool = True
    ):
        self.log_metric(
            f"{operation}_duration",
            duration,
            unit="seconds",
            tags={'operation': operation, 'success': str(success)}
        )


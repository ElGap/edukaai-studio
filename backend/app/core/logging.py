"""
Centralized Logging System for EdukaAI Studio

Features:
- Main log file recreated on every startup
- Separate error and debug log files
- Log rotation to prevent disk overflow
- Structured logging with timestamps
- Automatic directory management
"""

import logging
import logging.handlers
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Log directory paths
LOG_DIR = Path("./storage/app/logs")
MAIN_LOG = LOG_DIR / "main.log"
ERROR_LOG = LOG_DIR / "error.log"
DEBUG_LOG = LOG_DIR / "debug.log"
ARCHIVE_DIR = LOG_DIR / "archive"

# Maximum log file size (10MB)
MAX_BYTES = 10 * 1024 * 1024
# Number of backup files to keep
BACKUP_COUNT = 5


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Setup centralized logging system.
    
    This function:
    1. Creates log directory structure
    2. Archives existing main.log if it exists
    3. Creates new main.log
    4. Sets up file handlers with rotation
    5. Configures formatters
    6. Returns the root logger
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Root logger instance
    """
    # Create log directories
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Archive existing main.log if it exists
    if MAIN_LOG.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"main_{timestamp}.log"
        archive_path = ARCHIVE_DIR / archive_name
        
        try:
            shutil.move(str(MAIN_LOG), str(archive_path))
            print(f"[LOG] Archived existing log to: {archive_path}")
        except Exception as e:
            print(f"[LOG] Warning: Could not archive existing log: {e}")
    
    # Clean up old archive files (keep last 10)
    try:
        archive_files = sorted(ARCHIVE_DIR.glob("main_*.log"), key=os.path.getmtime)
        if len(archive_files) > 10:
            for old_file in archive_files[:-10]:
                old_file.unlink()
                print(f"[LOG] Removed old archive: {old_file.name}")
    except Exception as e:
        print(f"[LOG] Warning: Could not clean up old archives: {e}")
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Main log handler (all levels)
    main_handler = logging.handlers.RotatingFileHandler(
        MAIN_LOG,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding='utf-8'
    )
    main_handler.setLevel(logging.DEBUG)
    main_handler.setFormatter(formatter)
    root_logger.addHandler(main_handler)
    
    # Error log handler (errors only)
    error_handler = logging.handlers.RotatingFileHandler(
        ERROR_LOG,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)
    
    # Debug log handler (debug and above)
    debug_handler = logging.handlers.RotatingFileHandler(
        DEBUG_LOG,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding='utf-8'
    )
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(formatter)
    root_logger.addHandler(debug_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("EdukaAI Studio Logging System Initialized")
    logger.info(f"Main log: {MAIN_LOG}")
    logger.info(f"Error log: {ERROR_LOG}")
    logger.info(f"Debug log: {DEBUG_LOG}")
    logger.info(f"Archive dir: {ARCHIVE_DIR}")
    logger.info(f"Log level: {log_level}")
    logger.info("=" * 80)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Global exception hook for uncaught exceptions
def setup_exception_logging():
    """Setup global exception hook to log all uncaught exceptions."""
    logger = logging.getLogger("exceptions")
    
    def handle_exception(exc_type, exc_value, exc_traceback):
        """Log uncaught exceptions."""
        if issubclass(exc_type, KeyboardInterrupt):
            # Don't log keyboard interrupt
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger.critical(
            f"Uncaught exception: {exc_type.__name__}: {exc_value}",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
    
    sys.excepthook = handle_exception
    logger.info("Global exception handler installed")


class LogContext:
    """Context manager for adding context to logs."""
    
    def __init__(self, logger: logging.Logger, context: str):
        self.logger = logger
        self.context = context
        self.old_formatters = []
    
    def __enter__(self):
        for handler in self.logger.handlers:
            old_formatter = handler.formatter
            self.old_formatters.append((handler, old_formatter))
            
            new_format = f"%(asctime)s | %(levelname)-8s | %(name)s | [{self.context}] %(message)s"
            handler.setFormatter(logging.Formatter(new_format))
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for handler, old_formatter in self.old_formatters:
            handler.setFormatter(old_formatter)


# Initialize logging on module import
logger = get_logger(__name__)

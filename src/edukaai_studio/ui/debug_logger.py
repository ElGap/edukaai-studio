"""
Comprehensive Debug Logger for Training

This module provides detailed debug logging for training operations.
Logs are written to a file (not shown to users) for post-mortem analysis.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


# Create debug logger
debug_logger = logging.getLogger('training_debug')
debug_logger.setLevel(logging.DEBUG)

# Create file handler (logs to file, not console)
log_file = Path(__file__).parent.parent.parent / "training_debug.log"
file_handler = logging.FileHandler(log_file, mode='w')
file_handler.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S.%f'
)
file_handler.setFormatter(formatter)

# Add handler to logger
debug_logger.addHandler(file_handler)


def log_debug(msg):
    """Log debug message to file only."""
    debug_logger.debug(msg)


def log_info(msg):
    """Log info message to file only."""
    debug_logger.info(msg)


def log_warning(msg):
    """Log warning message to file only."""
    debug_logger.warning(msg)


def log_error(msg, exc_info=False):
    """Log error message to file only."""
    if exc_info:
        debug_logger.exception(msg)
    else:
        debug_logger.error(msg)


def log_subprocess_start(cmd, env_vars, cwd=None):
    """Log subprocess startup details."""
    log_info("=" * 80)
    log_info("SUBPROCESS START")
    log_info("=" * 80)
    log_info(f"Command: {' '.join(str(c) for c in cmd)}")
    log_info(f"Working directory: {cwd or Path.cwd()}")
    log_info(f"Environment variables:")
    for key, value in env_vars.items():
        # Mask sensitive values
        if 'token' in key.lower() or 'key' in key.lower() or 'secret' in key.lower():
            log_info(f"  {key}: [MASKED]")
        else:
            log_info(f"  {key}: {value}")
    log_info("-" * 80)


def log_subprocess_output(line, is_stderr=False):
    """Log subprocess output line."""
    prefix = "[STDERR]" if is_stderr else "[STDOUT]"
    log_debug(f"{prefix} {line}")


def log_subprocess_exit(returncode, lines_read):
    """Log subprocess exit details."""
    log_info("=" * 80)
    log_info("SUBPROCESS EXIT")
    log_info("=" * 80)
    log_info(f"Return code: {returncode}")
    log_info(f"Total lines read: {lines_read}")
    log_info("-" * 80)


def log_state_change(old_state, new_state, context=""):
    """Log state changes."""
    log_debug(f"State change {context}:")
    for key in set(old_state.keys()) | set(new_state.keys()):
        old_val = old_state.get(key, "<missing>")
        new_val = new_state.get(key, "<missing>")
        if old_val != new_val:
            log_debug(f"  {key}: {old_val} → {new_val}")


def log_exception(context):
    """Log exception with context."""
    log_error(f"Exception in {context}", exc_info=True)

"""
World-Class Logging Configuration for EdukaAI Studio

Production-ready logging with loguru featuring:
- Multiple log levels (TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL)
- Structured JSON logging for machine parsing
- Console output with beautiful colors
- File rotation and retention
- Contextual logging with bind()
- Configurable via environment variables
- Async-safe for Gradio

Environment Variables:
    LOG_LEVEL: Minimum log level (default: INFO)
    LOG_FORMAT: Console format (default: colored, options: colored, plain, json)
    LOG_FILE: Enable file logging (default: true)
    LOG_FILE_PATH: Log file location (default: logs/edukaai.log)
    LOG_FILE_ROTATION: Rotation size (default: 100 MB)
    LOG_FILE_RETENTION: Retention period (default: 30 days)
    LOG_STRUCTURED: Enable structured JSON logging (default: false)

Usage:
    from edukaai_studio.core.logging import get_logger, LogConfig
    
    # Simple usage
    logger = get_logger("ui.train")
    logger.info("Starting training")
    logger.error("Something went wrong")
    
    # With context
    logger = get_logger("ui.train").bind(iteration=50, model="phi-3")
    logger.info("Training progress")  # Automatically includes iteration and model
    
    # Structured logging for analytics
    logger.bind(structured=True).info("training_step", {
        "iteration": 50,
        "loss": 1.234,
        "speed": 3.5
    })
"""

import os
import sys
import logging
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field

# Try to import loguru, fallback to standard logging if not available
try:
    from loguru import logger as _logger
    LOGURU_AVAILABLE = True
except ImportError:
    LOGURU_AVAILABLE = False
    # Create a dummy logger that will be replaced
    _logger = None


class LogLevel(str, Enum):
    """Log levels ordered by severity."""
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogConfig:
    """
    Centralized logging configuration.
    
    All settings are configurable via environment variables for
    production deployments without code changes.
    """
    
    # Console settings
    LOG_LEVEL: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper())
    """Minimum log level to display (TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL)"""
    
    LOG_FORMAT: str = field(default_factory=lambda: os.getenv("LOG_FORMAT", "colored"))
    """Console format: colored, plain, or minimal"""
    
    LOG_COLORIZE: bool = field(default_factory=lambda: os.getenv("LOG_COLORIZE", "true").lower() == "true")
    """Enable colored console output"""
    
    # File settings
    LOG_FILE_ENABLE: bool = field(default_factory=lambda: os.getenv("LOG_FILE", "true").lower() == "true")
    """Enable file logging"""
    
    LOG_FILE_PATH: str = field(default_factory=lambda: os.getenv("LOG_FILE_PATH", "logs/edukaai.log"))
    """Log file path (relative to project root)"""
    
    LOG_FILE_ROTATION: str = field(default_factory=lambda: os.getenv("LOG_FILE_ROTATION", "100 MB"))
    """Rotate file when it reaches this size"""
    
    LOG_FILE_RETENTION: str = field(default_factory=lambda: os.getenv("LOG_FILE_RETENTION", "30 days"))
    """Keep logs for this period"""
    
    LOG_FILE_COMPRESSION: str = field(default_factory=lambda: os.getenv("LOG_FILE_COMPRESSION", "zip"))
    """Compress rotated logs (zip, gz, or none)"""
    
    # Structured logging
    LOG_STRUCTURED: bool = field(default_factory=lambda: os.getenv("LOG_STRUCTURED", "false").lower() == "true")
    """Enable structured JSON logging to separate file"""
    
    LOG_STRUCTURED_PATH: str = field(default_factory=lambda: os.getenv("LOG_STRUCTURED_PATH", "logs/edukaai.jsonl"))
    """Path for structured JSON logs"""
    
    # Component filtering
    LOG_INCLUDE_COMPONENTS: Optional[str] = field(default_factory=lambda: os.getenv("LOG_INCLUDE_COMPONENTS"))
    """Comma-separated list of components to include (e.g., "ui.train,core.state")"""
    
    LOG_EXCLUDE_COMPONENTS: Optional[str] = field(default_factory=lambda: os.getenv("LOG_EXCLUDE_COMPONENTS"))
    """Comma-separated list of components to exclude"""
    
    # Gradio integration
    LOG_GRADIO_SUPPRESS: bool = field(default_factory=lambda: os.getenv("LOG_GRADIO_SUPPRESS", "true").lower() == "true")
    """Suppress Gradio's default logging"""
    
    # Development features
    LOG_BACKTRACE: bool = field(default_factory=lambda: os.getenv("LOG_BACKTRACE", "false").lower() == "true")
    """Enable exception backtraces (development only, can leak secrets)"""
    
    LOG_DIAGNOSE: bool = field(default_factory=lambda: os.getenv("LOG_DIAGNOSE", "false").lower() == "true")
    """Enable exception diagnosis (development only)"""
    
    LOG_CATCH: bool = field(default_factory=lambda: os.getenv("LOG_CATCH", "true").lower() == "true")
    """Auto-catch exceptions in decorated functions"""


class LoggerManager:
    """
    Manages loguru configuration and provides logger instances.
    
    This is a singleton that ensures logging is configured once
    and provides filtered loggers for each component.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.config = LogConfig()
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure loguru with production-ready settings."""
        if not LOGURU_AVAILABLE:
            # Fallback to standard logging if loguru not available
            self._setup_fallback_logging()
            return
        
        # Remove default handler
        _logger.remove()
        
        # Setup console handler
        self._setup_console_handler()
        
        # Setup file handler
        if self.config.LOG_FILE_ENABLE:
            self._setup_file_handler()
        
        # Setup structured logging
        if self.config.LOG_STRUCTURED:
            self._setup_structured_handler()
        
        # Suppress noisy libraries
        if self.config.LOG_GRADIO_SUPPRESS:
            self._suppress_noisy_libraries()
    
    def _setup_fallback_logging(self):
        """Setup standard library logging as fallback."""
        # Configure basic logging
        logging.basicConfig(
            level=getattr(logging, self.config.LOG_LEVEL, logging.INFO),
            format='%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Create a logger that mimics loguru interface
        self._fallback_logger = logging.getLogger("edukaai")
    
    def _setup_console_handler(self):
        """Configure beautiful console output."""
        # Determine format based on configuration
        if self.config.LOG_FORMAT == "colored":
            format_string = (
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{extra[component]: <20}</cyan> | "
                "{extra[context]}"
                "<level>{message}</level>"
            )
        elif self.config.LOG_FORMAT == "plain":
            format_string = (
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level: <8} | "
                "{extra[component]: <20} | "
                "{extra[context]}"
                "{message}"
            )
        elif self.config.LOG_FORMAT == "minimal":
            format_string = "{time:HH:mm:ss} | {level: <8} | {message}"
        else:
            format_string = self.config.LOG_FORMAT
        
        # Add console handler with filtering
        _logger.add(
            sys.stderr,
            format=format_string,
            level=self.config.LOG_LEVEL,
            colorize=self.config.LOG_COLORIZE and self.config.LOG_FORMAT == "colored",
            filter=self._make_component_filter(),
            backtrace=self.config.LOG_BACKTRACE,
            diagnose=self.config.LOG_DIAGNOSE,
            enqueue=True,  # Thread-safe for Gradio
            catch=True,
        )
    
    def _setup_file_handler(self):
        """Configure rotating file logging."""
        # Ensure log directory exists
        log_path = Path(self.config.LOG_FILE_PATH)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        _logger.add(
            str(log_path),
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[component]: <20} | {message}",
            level="DEBUG",  # Always log everything to file
            rotation=self.config.LOG_FILE_ROTATION,
            retention=self.config.LOG_FILE_RETENTION,
            compression=self.config.LOG_FILE_COMPRESSION,
            encoding="utf-8",
            filter=self._make_component_filter(),
            backtrace=True,  # Always enable in files
            diagnose=True,
            enqueue=True,
        )
    
    def _setup_structured_handler(self):
        """Configure structured JSON logging for analytics."""
        structured_path = Path(self.config.LOG_STRUCTURED_PATH)
        structured_path.parent.mkdir(parents=True, exist_ok=True)
        
        def json_formatter(record):
            """Format log record as JSON with structured data."""
            import json
            from datetime import datetime
            
            log_entry = {
                "timestamp": record["time"].isoformat(),
                "level": record["level"].name,
                "component": record["extra"].get("component", "unknown"),
                "message": record["message"],
                "context": record["extra"].get("context_data", {}),
            }
            
            # Add exception info if present
            if record["exception"]:
                log_entry["exception"] = {
                    "type": record["exception"].type.__name__ if record["exception"].type else None,
                    "value": str(record["exception"].value) if record["exception"].value else None,
                }
            
            return json.dumps(log_entry, default=str) + "\n"
        
        _logger.add(
            str(structured_path),
            format=json_formatter,
            level="INFO",
            rotation=self.config.LOG_FILE_ROTATION,
            retention=self.config.LOG_FILE_RETENTION,
            compression=self.config.LOG_FILE_COMPRESSION,
            encoding="utf-8",
            filter=lambda record: record["extra"].get("structured", False),
            enqueue=True,
        )
    
    def _make_component_filter(self):
        """Create a filter function based on include/exclude lists."""
        include = set()
        exclude = set()
        
        if self.config.LOG_INCLUDE_COMPONENTS:
            include = set(self.config.LOG_INCLUDE_COMPONENTS.split(","))
        
        if self.config.LOG_EXCLUDE_COMPONENTS:
            exclude = set(self.config.LOG_EXCLUDE_COMPONENTS.split(","))
        
        def filter_func(record):
            component = record["extra"].get("component", "")
            
            # If include list specified, only include those components
            if include and component not in include:
                return False
            
            # Always exclude specified components
            if component in exclude:
                return False
            
            return True
        
        return filter_func
    
    def _suppress_noisy_libraries(self):
        """Suppress logging from noisy third-party libraries."""
        import logging
        
        # Suppress common noisy libraries
        noisy_loggers = [
            "gradio",
            "httpx",
            "urllib3",
            "matplotlib",
            "PIL",
        ]
        
        for name in noisy_loggers:
            logging.getLogger(name).setLevel(logging.WARNING)
    
    def get_logger(self, component: str, **context):
        """
        Get a logger instance bound to a component.
        
        Args:
            component: Component name (e.g., "ui.train", "core.state")
            **context: Default context values to include in all logs
            
        Returns:
            Logger instance with component binding
        """
        if not LOGURU_AVAILABLE:
            # Return fallback logger
            return FallbackComponentLogger(self._fallback_logger, component, context)
        
        # Create context string for display
        context_str = ""
        if context:
            context_parts = [f"{k}={v}" for k, v in context.items()]
            context_str = "[" + ", ".join(context_parts) + "] "
        
        # Bind component and context
        bound_logger = _logger.bind(
            component=component,
            context=context_str,
            context_data=context,
            structured=False,
        )
        
        return ComponentLogger(bound_logger, component, context)
    
    def update_config(self, **kwargs):
        """Update logging configuration at runtime."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Re-setup logging with new config
        self._setup_logging()


class ComponentLogger:
    """
    Wrapper around loguru logger with component-specific features.
    
    Provides convenient methods for contextual logging and structured
    data output.
    """
    
    def __init__(self, logger, component: str, context: Dict[str, Any]):
        self._logger = logger
        self.component = component
        self.context = context
    
    def bind(self, **kwargs):
        """Create new logger with additional context."""
        new_context = {**self.context, **kwargs}
        return LoggerManager().get_logger(self.component, **new_context)
    
    def structured(self):
        """Enable structured JSON logging for this logger."""
        return ComponentLogger(
            self._logger.bind(structured=True),
            self.component,
            self.context
        )
    
    # Standard logging methods
    def trace(self, message: str, *args, **kwargs):
        """Log at TRACE level (very detailed, development only)."""
        self._logger.trace(message, *args, **kwargs)
    
    def debug(self, message: str, *args, **kwargs):
        """Log at DEBUG level (detailed information)."""
        self._logger.debug(message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """Log at INFO level (general information)."""
        self._logger.info(message, *args, **kwargs)
    
    def success(self, message: str, *args, **kwargs):
        """Log at SUCCESS level (positive outcomes)."""
        self._logger.success(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """Log at WARNING level (potential issues)."""
        self._logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """Log at ERROR level (recoverable errors)."""
        self._logger.error(message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """Log at CRITICAL level (system failures)."""
        self._logger.critical(message, *args, **kwargs)
    
    def exception(self, message: str, *args, **kwargs):
        """Log exception with traceback."""
        self._logger.exception(message, *args, **kwargs)
    
    def catch(self, level="ERROR", reraise=True, message="An error occurred"):
        """Decorator to catch and log exceptions."""
        return self._logger.catch(level=level, reraise=reraise, message=message)
    
    # Event logging helper
    def event(self, event_name: str, data: Optional[Dict[str, Any]] = None, level="INFO"):
        """
        Log a structured event with optional data.
        
        Args:
            event_name: Name of the event (e.g., "training_started")
            data: Optional dictionary of event data
            level: Log level for the event
        """
        if data:
            formatted_data = " | ".join([f"{k}={v}" for k, v in data.items()])
            message = f"[{event_name}] {formatted_data}"
        else:
            message = f"[{event_name}]"
        
        log_func = getattr(self._logger, level.lower())
        log_func(message)
    
    def progress(self, current: int, total: int, message: str = "", **extra_data):
        """
        Log training progress in a standardized format.
        
        Args:
            current: Current iteration
            total: Total iterations
            message: Optional message
            **extra_data: Additional metrics (loss, speed, etc.)
        """
        pct = (current / total * 100) if total > 0 else 0
        
        parts = [f"iter={current}/{total} ({pct:.1f}%)"]
        
        if message:
            parts.append(message)
        
        for key, value in extra_data.items():
            parts.append(f"{key}={value}")
        
        self._logger.info(" | ".join(parts))
    
    def metric(self, name: str, value: Union[int, float, str], unit: str = ""):
        """
        Log a single metric value.
        
        Args:
            name: Metric name
            value: Metric value
            unit: Optional unit (GB, it/s, etc.)
        """
        if unit:
            self._logger.info(f"metric: {name}={value}{unit}")
        else:
            self._logger.info(f"metric: {name}={value}")
    
    def state_change(self, from_state: str, to_state: str, reason: str = ""):
        """Log a state transition."""
        if reason:
            self._logger.info(f"state: {from_state} -> {to_state} ({reason})")
        else:
            self._logger.info(f"state: {from_state} -> {to_state}")


class FallbackComponentLogger:
    """
    Fallback logger when loguru is not available.
    
    Mimics ComponentLogger interface using standard library logging.
    """
    
    def __init__(self, logger, component: str, context: Dict[str, Any]):
        self._logger = logger
        self.component = component
        self.context = context
    
    def bind(self, **kwargs):
        """Create new logger with additional context."""
        new_context = {**self.context, **kwargs}
        return FallbackComponentLogger(self._logger, self.component, new_context)
    
    def structured(self):
        """No-op for fallback (structured logging requires loguru)."""
        return self
    
    def _format_message(self, message: str) -> str:
        """Format message with context."""
        if self.context:
            context_str = " | ".join([f"{k}={v}" for k, v in self.context.items()])
            return f"[{self.component}] [{context_str}] {message}"
        return f"[{self.component}] {message}"
    
    def trace(self, message: str, *args, **kwargs):
        """Log at TRACE level (maps to DEBUG in standard logging)."""
        self._logger.debug(self._format_message(message), *args, **kwargs)
    
    def debug(self, message: str, *args, **kwargs):
        """Log at DEBUG level."""
        self._logger.debug(self._format_message(message), *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """Log at INFO level."""
        self._logger.info(self._format_message(message), *args, **kwargs)
    
    def success(self, message: str, *args, **kwargs):
        """Log at SUCCESS level (maps to INFO)."""
        self._logger.info(self._format_message(f"✓ {message}"), *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """Log at WARNING level."""
        self._logger.warning(self._format_message(message), *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """Log at ERROR level."""
        self._logger.error(self._format_message(message), *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """Log at CRITICAL level."""
        self._logger.critical(self._format_message(message), *args, **kwargs)
    
    def exception(self, message: str, *args, **kwargs):
        """Log exception with traceback."""
        self._logger.exception(self._format_message(message), *args, **kwargs)
    
    def catch(self, level="ERROR", reraise=True, message="An error occurred"):
        """Decorator to catch and log exceptions."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self.exception(f"{message}: {e}")
                    if reraise:
                        raise
            return wrapper
        return decorator
    
    def event(self, event_name: str, data: Optional[Dict[str, Any]] = None, level="INFO"):
        """Log a structured event."""
        if data:
            formatted_data = " | ".join([f"{k}={v}" for k, v in data.items()])
            message = f"[{event_name}] {formatted_data}"
        else:
            message = f"[{event_name}]"
        
        log_func = getattr(self, level.lower())
        log_func(message)
    
    def progress(self, current: int, total: int, message: str = "", **extra_data):
        """Log training progress."""
        pct = (current / total * 100) if total > 0 else 0
        parts = [f"iter={current}/{total} ({pct:.1f}%)"]
        if message:
            parts.append(message)
        for key, value in extra_data.items():
            parts.append(f"{key}={value}")
        self.info(" | ".join(parts))
    
    def metric(self, name: str, value: Union[int, float, str], unit: str = ""):
        """Log a single metric value."""
        if unit:
            self.info(f"metric: {name}={value}{unit}")
        else:
            self.info(f"metric: {name}={value}")
    
    def state_change(self, from_state: str, to_state: str, reason: str = ""):
        """Log a state transition."""
        if reason:
            self.info(f"state: {from_state} -> {to_state} ({reason})")
        else:
            self.info(f"state: {from_state} -> {to_state}")


# Convenience functions
def get_logger(component: str, **context):
    """
    Get a logger instance for a component.
    
    This is the main entry point for logging in the application.
    
    Args:
        component: Component identifier (e.g., "ui.train", "core.state")
        **context: Default context values for all log messages
        
    Returns:
        ComponentLogger or FallbackComponentLogger instance
        
    Example:
        >>> from edukaai_studio.core.logging import get_logger
        >>> logger = get_logger("ui.train")
        >>> logger.info("Training started")
        >>> 
        >>> # With context
        >>> logger = get_logger("ui.train", model="phi-3", iteration=50)
        >>> logger.info("Progress update")  # Shows: model=phi-3, iteration=50
    """
    return LoggerManager().get_logger(component, **context)


def configure_logging(**kwargs):
    """
    Update logging configuration at runtime.
    
    Args:
        **kwargs: Configuration options (see LogConfig)
        
    Example:
        >>> configure_logging(LOG_LEVEL="DEBUG", LOG_FILE_ENABLE=True)
    """
    LoggerManager().update_config(**kwargs)


def get_log_config() -> LogConfig:
    """Get current logging configuration."""
    return LoggerManager().config


# Module-level logger for this module itself
_module_logger = get_logger("core.logging")


def log_system_startup():
    """Log system startup with configuration summary."""
    config = get_log_config()
    
    _module_logger.info("=" * 70)
    _module_logger.info("EDUKAAI STUDIO - SYSTEM STARTUP")
    _module_logger.info("=" * 70)
    
    _module_logger.info(f"Log Level: {config.LOG_LEVEL}")
    _module_logger.info(f"Console Format: {config.LOG_FORMAT}")
    _module_logger.info(f"File Logging: {config.LOG_FILE_ENABLE}")
    
    if config.LOG_FILE_ENABLE:
        _module_logger.info(f"Log File: {config.LOG_FILE_PATH}")
        _module_logger.info(f"Rotation: {config.LOG_FILE_ROTATION}")
        _module_logger.info(f"Retention: {config.LOG_FILE_RETENTION}")
    
    if config.LOG_STRUCTURED:
        _module_logger.info(f"Structured Logging: {config.LOG_STRUCTURED_PATH}")
    
    if config.LOG_INCLUDE_COMPONENTS:
        _module_logger.info(f"Include Components: {config.LOG_INCLUDE_COMPONENTS}")
    
    if config.LOG_EXCLUDE_COMPONENTS:
        _module_logger.info(f"Exclude Components: {config.LOG_EXCLUDE_COMPONENTS}")
    
    _module_logger.info("=" * 70)


# Initialize on import
LoggerManager()

"""
Logging Usage Examples for EdukaAI Studio

This file demonstrates how to use the world-class logging system.
Run with: python examples/logging_examples.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from edukaai_studio.core import get_logger, configure_logging


def example_basic_logging():
    """Basic logging with different levels."""
    logger = get_logger("example.basic")
    
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Logging")
    print("=" * 70)
    
    logger.trace("Very detailed trace message (development only)")
    logger.debug("Debug information for troubleshooting")
    logger.info("General information message")
    logger.success("Success! Operation completed")
    logger.warning("Warning: something might be wrong")
    logger.error("Error: something went wrong")
    logger.critical("Critical: system failure")


def example_contextual_logging():
    """Logging with automatic context."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Contextual Logging")
    print("=" * 70)
    
    # Bind context at creation
    logger = get_logger("example.context", model="phi-3", iteration=50)
    
    logger.info("Training started")
    logger.info("Processing batch")
    
    # Add more context for a subsection
    subsection_logger = logger.bind(batch_size=32)
    subsection_logger.info("Batch processed")


def example_event_logging():
    """Structured event logging."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Event Logging")
    print("=" * 70)
    
    logger = get_logger("example.events")
    
    # Log training lifecycle events
    logger.event("training_initiated", {
        "model": "mlx-community/Phi-3-mini-4k-instruct-4bit",
        "iterations": 600,
        "learning_rate": "1e-4",
        "lora_rank": 16
    })
    
    logger.event("training_progress", {
        "iteration": 100,
        "train_loss": 1.234,
        "val_loss": 1.345,
        "speed": 3.5
    })
    
    logger.event("training_complete", {
        "final_loss": 1.123,
        "best_iteration": 450,
        "duration_minutes": 45
    })


def example_progress_logging():
    """Standardized progress logging."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Progress Logging")
    print("=" * 70)
    
    logger = get_logger("example.progress")
    
    # Simulate training progress
    for i in range(0, 201, 50):
        logger.progress(
            current=i,
            total=200,
            loss=2.5 - (i / 200) * 1.5,  # Simulated decreasing loss
            speed=3.2 + (i % 3) * 0.1,   # Simulated speed variation
            memory_gb=4.0 + (i / 200)    # Simulated memory growth
        )


def example_metric_logging():
    """Individual metric logging."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Metric Logging")
    print("=" * 70)
    
    logger = get_logger("example.metrics")
    
    logger.metric("train_loss", 1.234)
    logger.metric("val_loss", 1.345)
    logger.metric("memory_usage", 4.5, unit="GB")
    logger.metric("training_speed", 3.2, unit="it/s")
    logger.metric("peak_memory", 14.2, unit="GB")


def example_state_changes():
    """State transition logging."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: State Changes")
    print("=" * 70)
    
    logger = get_logger("example.state")
    
    # Training lifecycle states
    logger.state_change("idle", "initializing", "User clicked start")
    logger.state_change("initializing", "loading_model", "Downloading from HuggingFace")
    logger.state_change("loading_model", "training", "Model loaded successfully")
    logger.state_change("training", "validating", "Iteration 50 reached")
    logger.state_change("validating", "training", "Validation complete")
    logger.state_change("training", "complete", "All iterations finished")
    logger.state_change("complete", "saving", "Exporting adapter")
    logger.state_change("saving", "idle", "Training session complete")


def example_exception_handling():
    """Exception handling with automatic logging."""
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Exception Handling")
    print("=" * 70)
    
    logger = get_logger("example.exceptions")
    
    # Method 1: Decorator
    @logger.catch(reraise=False)
    def risky_operation():
        raise ValueError("Something went wrong!")
    
    risky_operation()
    
    # Method 2: Manual try/except
    try:
        1 / 0
    except Exception:
        logger.exception("Division by zero occurred")


def example_component_filtering():
    """Demonstrate component filtering."""
    print("\n" + "=" * 70)
    print("EXAMPLE 8: Component Filtering")
    print("=" * 70)
    
    # Create loggers for different components
    train_logger = get_logger("ui.train")
    chat_logger = get_logger("ui.chat")
    state_logger = get_logger("core.state")
    monitor_logger = get_logger("monitor.progress")
    
    train_logger.info("Training log message")
    chat_logger.info("Chat log message")
    state_logger.info("State log message")
    monitor_logger.info("Monitor log message")
    
    print("\nTo filter components, set environment variable:")
    print("export LOG_INCLUDE_COMPONENTS='ui.train,core.state'")
    print("export LOG_EXCLUDE_COMPONENTS='ui.chat'")


def example_runtime_configuration():
    """Change logging configuration at runtime."""
    print("\n" + "=" * 70)
    print("EXAMPLE 9: Runtime Configuration")
    print("=" * 70)
    
    logger = get_logger("example.config")
    
    print("\n--- Before configuration change ---")
    logger.debug("This won't appear (default level is INFO)")
    logger.info("This will appear")
    
    # Change to DEBUG level
    configure_logging(LOG_LEVEL="DEBUG")
    
    print("\n--- After changing to DEBUG level ---")
    logger.debug("Now this appears!")
    logger.info("This still appears")


def example_structured_logging():
    """Structured JSON logging for analytics."""
    print("\n" + "=" * 70)
    print("EXAMPLE 10: Structured Logging")
    print("=" * 70)
    
    logger = get_logger("example.structured")
    
    # Enable structured logging
    configure_logging(LOG_STRUCTURED=True)
    
    # Use structured() method for JSON output
    structured_logger = logger.structured()
    
    structured_logger.info("training_step", {
        "iteration": 50,
        "train_loss": 1.234,
        "val_loss": 1.345,
        "learning_rate": "1e-4",
        "speed": 3.5,
        "memory_gb": 4.2
    })
    
    structured_logger.info("validation_result", {
        "iteration": 50,
        "val_loss": 1.345,
        "improvement": 0.089
    })
    
    print("\nCheck logs/edukaai.jsonl for structured output")


def example_real_world_training():
    """Real-world training scenario."""
    print("\n" + "=" * 70)
    print("EXAMPLE 11: Real-World Training Scenario")
    print("=" * 70)
    
    # Configure for production-like output
    configure_logging(
        LOG_LEVEL="INFO",
        LOG_FORMAT="colored"
    )
    
    # Main training logger with initial context
    logger = get_logger(
        "ui.train",
        model="phi-3-mini",
        data_file="alpaca_clean.jsonl",
        iterations=200
    )
    
    # Training lifecycle
    logger.success("Training session initiated")
    
    logger.event("config", {
        "model": "mlx-community/Phi-3-mini-4k-instruct-4bit",
        "iterations": 200,
        "learning_rate": "1e-4",
        "lora_rank": 16,
        "lora_alpha": 32,
        "grad_accumulation": 32
    })
    
    logger.state_change("idle", "initializing")
    logger.info("Loading model from HuggingFace...")
    logger.success("Model loaded successfully (3.8B parameters)")
    
    logger.state_change("initializing", "training")
    logger.info("Starting training loop")
    
    # Simulate training iterations
    train_losses = [2.5, 2.1, 1.8, 1.5, 1.3, 1.2, 1.15, 1.12, 1.10, 1.08]
    val_losses = [2.4, 2.0, 1.75, 1.55, 1.35, 1.25, 1.20, 1.18, 1.16, 1.15]
    
    for i in range(0, 200, 20):
        idx = min(i // 20, len(train_losses) - 1)
        
        logger.progress(
            current=i,
            total=200,
            loss=train_losses[idx],
            val_loss=val_losses[idx],
            speed=3.2 + (i % 5) * 0.1,
            memory_gb=4.0 + (i / 200) * 0.5,
            message=f"Training iter {i}"
        )
        
        # Validation checkpoint every 50 iterations
        if i % 50 == 0 and i > 0:
            logger.event("validation", {
                "iteration": i,
                "val_loss": val_losses[idx],
                "best_so_far": min(val_losses[:idx+1])
            })
    
    logger.state_change("training", "complete")
    logger.success(f"Training complete! Final loss: {train_losses[-1]:.4f}")
    
    logger.metric("final_train_loss", train_losses[-1])
    logger.metric("final_val_loss", val_losses[-1])
    logger.metric("training_duration", 25, unit="min")
    logger.metric("peak_memory", 4.5, unit="GB")
    
    logger.state_change("complete", "saving")
    logger.info("Saving adapter weights...")
    logger.success("Adapter saved to outputs/phi3_lora_20260323/")
    
    logger.state_change("saving", "idle")
    logger.success("Training session complete! Check Results tab for downloads.")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("EDUKAAI STUDIO - LOGGING SYSTEM EXAMPLES")
    print("=" * 70)
    
    # Run all examples
    example_basic_logging()
    example_contextual_logging()
    example_event_logging()
    example_progress_logging()
    example_metric_logging()
    example_state_changes()
    example_exception_handling()
    example_component_filtering()
    example_runtime_configuration()
    # example_structured_logging()  # Uncomment to test structured logging
    example_real_world_training()
    
    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETE")
    print("=" * 70)
    print("\nFor more information, see docs/LOGGING.md")

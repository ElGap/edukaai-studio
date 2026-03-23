"""
UI Package for LoRA Training Studio

Provides modules for the desktop UI wrapper.
"""

from .training_monitor import TrainingMonitor
from .chat_wrapper import ChatWrapper
try:
    from .visualizations import create_loss_chart, format_metrics_summary
except ImportError:
    # Plotly not available
    create_loss_chart = None
    format_metrics_summary = None

__all__ = [
    'TrainingMonitor',
    'ChatWrapper', 
    'create_loss_chart',
    'format_metrics_summary'
]

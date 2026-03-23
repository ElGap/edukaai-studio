"""UI Tabs for EdukaAI Studio."""

from .upload import create_upload_tab
from .configure import create_configure_tab
from .train import create_train_tab
from .results import create_results_tab
from .chat import create_chat_tab
from .models import create_models_tab
from .my_models import create_my_models_tab

__all__ = [
    'create_upload_tab',
    'create_configure_tab',
    'create_train_tab',
    'create_results_tab',
    'create_chat_tab',
    'create_models_tab',
    'create_my_models_tab'
]
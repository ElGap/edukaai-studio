"""
Core exceptions for EdukaAI Studio.
"""


class EdukaAIException(Exception):
    """Base exception for EdukaAI Studio."""
    
    def __init__(
        self, 
        detail: str, 
        status_code: int = 500, 
        error_code: str = "internal_error"
    ):
        self.detail = detail
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(self.detail)


class ValidationError(EdukaAIException):
    """Validation error."""
    
    def __init__(self, detail: str):
        super().__init__(detail, status_code=422, error_code="validation_error")


class NotFoundError(EdukaAIException):
    """Resource not found error."""
    
    def __init__(self, detail: str = "Resource not found"):
        super().__init__(detail, status_code=404, error_code="not_found")


class ResourceLimitError(EdukaAIException):
    """Resource limit exceeded error."""
    
    def __init__(self, detail: str = "Resource limit exceeded"):
        super().__init__(detail, status_code=429, error_code="resource_limit")


class TrainingError(EdukaAIException):
    """Training execution error."""
    
    def __init__(self, detail: str = "Training failed"):
        super().__init__(detail, status_code=500, error_code="training_error")


class ExportError(EdukaAIException):
    """Model export error."""
    
    def __init__(self, detail: str = "Export failed"):
        super().__init__(detail, status_code=500, error_code="export_error")

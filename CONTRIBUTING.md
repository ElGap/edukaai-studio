# Contributing to EdukaAI Studio

Thank you for your interest in contributing to EdukaAI Studio! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork the repository
2. Clone your fork:
```bash
git clone https://github.com/YOUR_USERNAME/edukaai-studio.git
cd edukaai-studio
```

3. Create a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

4. Install development dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available
```

5. Run the application:
```bash
python edukaai-studio-ui.py
```

## Code Style Guidelines

- **Line length**: 100 characters maximum
- **Quotes**: Use double quotes for strings, triple double quotes for docstrings
- **Indentation**: 4 spaces (no tabs)
- **Type hints**: Use for function signatures and class attributes
- **Docstrings**: Use Google-style docstrings for all public functions and classes

### Import Order
```python
# 1. Standard library imports (alphabetical)
import json
import os
import sys
from datetime import datetime

# 2. Third-party imports (alphabetical)
import gradio as gr
import mlx
import numpy as np
from transformers import AutoTokenizer

# 3. Local imports (alphabetical)
from config import SERVER, STUDIO_MODELS
from ui.chat_wrapper import ChatWrapper
```

## Making Changes

1. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes following the code style guidelines

3. Test your changes thoroughly

4. Commit with a descriptive message:
```bash
git commit -m "Add feature: description of what was added"
```

## Pull Request Process

1. Update the README.md if needed with details of changes
2. Ensure your code follows the style guidelines
3. Create a Pull Request with a clear description
4. Reference any related issues

## Reporting Issues

When reporting issues, please include:

- macOS version
- Python version
- MLX version
- Steps to reproduce
- Expected behavior
- Actual behavior
- Error messages (if any)

## Code Review Checklist

- [ ] Type hints added to function signatures
- [ ] Docstrings for public APIs
- [ ] Error handling with specific exceptions
- [ ] No hardcoded values (use config.py)
- [ ] Imports sorted and grouped correctly
- [ ] Line length <= 100 characters
- [ ] No debug print statements left in production code
- [ ] Constants use UPPER_SNAKE_CASE

## Questions?

Feel free to open an issue for questions or join discussions.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

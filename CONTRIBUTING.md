# Contributing to EdukaAI Studio

First off, thank you for considering contributing to EdukaAI Studio! It's people like you that make this tool better for the Apple Silicon ML community.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Workflow](#development-workflow)
- [Style Guides](#style-guides)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project and everyone participating in it is governed by our commitment to:
- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Respect different viewpoints and experiences

## Getting Started

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.10+
- Node.js 18+
- Git

### Setting Up Development Environment

```bash
# 1. Fork and clone
git clone https://github.com/YOUR_USERNAME/edukai-studio.git
cd edukai-studio

# 2. Backend setup
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Frontend setup
cd ../frontend
npm install

# 4. Run tests to ensure everything works
cd ../backend
python -m pytest tests/ -v
```

## How Can I Contribute?

### Reporting Bugs

Before creating a bug report, please:
- Check if the issue already exists
- Try to reproduce with the latest version
- Include steps to reproduce
- Include system information (macOS version, Python version, etc.)
- Include relevant logs or error messages

Use the bug report template when creating an issue.

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating one:
- Use a clear, descriptive title
- Provide a detailed description of the proposed feature
- Explain why this enhancement would be useful
- List possible implementation approaches

### Pull Requests

1. Fork the repository
2. Create a new branch from `main`
3. Make your changes
4. Add or update tests as needed
5. Ensure all tests pass
6. Update documentation if needed
7. Submit a pull request

## Development Workflow

### Branch Naming

- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring

Example: `feature/add-gguf-export`

### Running the Application

```bash
# Terminal 1 - Backend
cd backend
source .venv/bin/activate
python run.py

# Terminal 2 - Frontend  
cd frontend
npm run dev
```

### Running Tests

```bash
# Backend tests
cd backend
python -m pytest tests/ -v

# Type checking (frontend)
cd frontend
npm run type-check

# Build verification
cd frontend
npm run build
```

## Style Guides

### Python (Backend)

- Follow PEP 8
- Use type hints where possible
- Maximum line length: 100 characters
- Use docstrings for functions and classes

Example:
```python
def process_dataset(dataset_id: str, config: DatasetConfig) -> Dataset:
    """
    Process and validate a dataset.
    
    Args:
        dataset_id: Unique identifier for the dataset
        config: Processing configuration
        
    Returns:
        Processed dataset object
    """
    # Implementation
```

### Vue.js/TypeScript (Frontend)

- Use Composition API with `<script setup>`
- Use TypeScript for type safety
- Follow Vue Style Guide
- Component names should be multi-word (except root App)

Example:
```vue
<script setup lang="ts">
import { ref, computed } from 'vue'

interface Props {
  modelId: string
}

const props = defineProps<Props>()
const isLoading = ref(false)

const formattedName = computed(() => {
  return props.modelId.toUpperCase()
})
</script>
```

### CSS/Tailwind

- Use Tailwind utility classes where possible
- Custom CSS only when necessary
- Maintain dark theme consistency
- Use semantic color names (avoid hardcoded hex values)

## Commit Messages

Use conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, missing semi-colons, etc)
- `refactor`: Code refactoring
- `test`: Adding or correcting tests
- `chore`: Build process or auxiliary tool changes

Examples:
```
feat(training): add support for custom learning rate schedules

fix(ui): resolve modal overflow on small screens

docs(readme): update installation instructions
```

## Pull Request Process

1. **Update documentation** - If your change affects how users use the tool, update the README.

2. **Add tests** - New features should include tests. Bug fixes should include regression tests.

3. **Ensure CI passes** - All tests must pass before merging.

4. **Request review** - At least one maintainer approval is required.

5. **Squash commits** - We'll squash commits on merge to keep history clean.

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe the tests you ran

## Screenshots (if applicable)
Add screenshots for UI changes

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Code follows style guide
- [ ] Self-review completed
```

## Areas for Contribution

### High Priority

1. **Testing Coverage** - Expand test suite
2. **Documentation** - Tutorials, examples, API docs
3. **UI/UX Improvements** - Better error messages, loading states

### Medium Priority

1. **Additional Models** - Support for new MLX-compatible models
2. **Training Presets** - Domain-specific presets (medical, legal, etc.)
3. **Export Formats** - Additional export options

### Nice to Have

1. **Performance** - Optimization for large datasets
2. **Features** - Data augmentation, advanced LoRA configs
3. **Tooling** - CLI tools, programmatic API

## Questions?

Feel free to:
- Open an issue with the "question" label
- Start a discussion on GitHub Discussions
- Contact maintainers directly (for maintainers: add your contact info)

Thank you for contributing! 🎉

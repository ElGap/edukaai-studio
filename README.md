# EdukaAI Studio

> Fine-tune Large Language Models on Apple Silicon with an intuitive web interface

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![macOS](https://img.shields.io/badge/macOS-Apple%20Silicon-blue.svg)](https://www.apple.com/macos/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MLX](https://img.shields.io/badge/MLX-Apple%20Silicon-orange.svg)](https://github.com/ml-explore/mlx)

EdukaAI Studio is a user-friendly application for fine-tuning Large Language Models (LLMs) on your Apple Silicon Mac. Built on Apple's [MLX framework](https://github.com/ml-explore/mlx), it provides a simple 5-step wizard to train custom AI models - no coding required!

![EdukaAI Studio Overview](/docs/screenshots/EdukaAi-Studio-training.png)

**Perfect for:**
- Students learning about AI/ML
- Professionals customizing models for specific domains
- Researchers experimenting with fine-tuning
- Hobbyists exploring LLM capabilities

## Quick Start (5 Minutes)

### Installation

**Recommended install**:
```bash
mkdir edukaai-studio
cd edukaai-studio
curl -fsSL https://raw.githubusercontent.com/elgap/edukaai-studio/main/install.sh | bash
```

**Non-interactive install** (for automation/CI):
```bash
mkdir edukaai-studio
cd edukaai-studio
curl -fsSL https://raw.githubusercontent.com/elgap/edukaai-studio/main/install.sh | bash -s -- --yes
```

The installer will:
- Check your Mac compatibility (Apple Silicon required)
- Install Python 3.10+ and Node.js 18+ (if missing)
- Download EdukaAI Studio to the current directory
- Install all dependencies
- Create a Desktop shortcut

### Launch the Application

After installation:
```bash
# Option 1: Double-click 'EdukaAI-Studio.command' on your Desktop
# Option 2: Run from the install directory:
./launch.sh
```

Then open your browser: **http://localhost:3030**

Check out the [Quick Start Guide](QUICKSTART.md) for a visual step-by-step walkthrough with screenshots!

## Your First Fine-Tuning (10 Minutes, no-code)

### Zero-Prep Option: EdukaAi Started Pack

Don't have training data? No problem! **Download EdukaAi starter pack** from [eduka.elgap.ai](https://eduka.elgap.ai) to get you going immediately.

#### EdukaAi Starter Pack includes: 
- real world data (JSONL) ready for fine tuning
- an AI-generated story (PDF)
- Basic instructions (PDF)

> ⚡ **Literally**: Download Starter Pack → Import to EdukaAi Studio → Click Configure → Click Train → Sip Coffee → Click Chat to test your fine tuned model in dual chat

### Manual Option: Create Your Own Data
Create a text file with training examples:

**Example 1 - Q&A Format:**
```
Question: What is machine learning?
Answer: Machine learning is a method of data analysis that automates analytical model building.

Question: How does a neural network work?
Answer: A neural network works by processing information through interconnected nodes organized in layers.
```

**Example 2 - Instruction Format:**
```json
{"instruction": "Translate to French", "input": "Hello, how are you?", "output": "Bonjour, comment allez-vous?"}
{"instruction": "Summarize this article", "input": "The quick brown fox...", "output": "A fox jumped over a lazy dog."}
```

Save as `my-training-data.jsonl` (100-1000 examples recommended)


### All-in-one option: **[AI Curator](https://github.com/ElGap/ai-curator)**
Your local-first data preparation layer. Import samples from various sources. Review, rate, create custom Q&A pairs, and craft the perfect dataset before exporting to EdukaAI Studio.

**Why Curator?**
- 🏠 **100% Local** — Your data never leaves your Mac
- 🔌 **Universal Import** — Capture API accepts data from any source (Kaggle, Hugging Face, APIs, files)
- ✏️ **Visual Editor** — Review samples and create custom question/answer pairs
- 🎯 **Studio Integration** — One-click export to EdukaAI Studio
- ☕ **Coffee-Break Speed** — From raw data to fine-tuned model in minutes

---

## Getting Help

**Quick Start Guide:** [QUICKSTART.md](QUICKSTART.md) - Visual step-by-step guide with screenshots

**Report Issues:** [GitHub Issues](https://github.com/elgap/edukaai-studio/issues)

**Discussions:** [GitHub Discussions](https://github.com/elgap/edukaai-studio/discussions)

---
## 🛠️ Development

### Tech Stack

**Backend:**
- FastAPI - High-performance async web framework
- MLX - Apple's machine learning framework
- SQLAlchemy - Database ORM
- Pydantic - Data validation
- Python-multipart - File uploads

**Frontend:**
- Vue.js 3 - Progressive JavaScript framework
- Pinia - State management
- Recharts - Data visualization
- Tailwind CSS - Utility-first CSS


**ML Libraries:**
- mlx-lm - Language model fine-tuning
- transformers - Model utilities
- safetensors - Efficient model storage

### Quick Setup for Developers

```bash
# Clone the repository
git clone https://github.com/elgap/edukaai-studio.git
cd edukaai-studio

# Run the installer (automatically detects developer mode)
./install.sh

# Start the application
./launch.sh
```

The installer will:
- Set up Python virtual environment
- Install all dependencies
- Run the test suite
- Create storage directories

### Project Structure

```
├── backend/
│   ├── app/
│   │   ├── routers/        # API endpoints
│   │   ├── ml/            # Training pipeline
│   │   ├── models.py      # Database schemas
│   │   └── main.py        # FastAPI app
│   ├── tests/             # Test suite
│   └── storage/           # Data storage
│
├── frontend/
│   ├── src/
│   │   ├── views/         # Page components
│   │   ├── components/    # Shared components
│   │   ├── stores/        # Pinia stores
│   │   └── router/        # Vue Router
│   └── public/           # Static assets
│
└── docs/                 # Documentation
```

### Running Tests

```bash
# Backend tests
cd backend
source .venv/bin/activate
python -m pytest tests/ -v

# Frontend build check
cd frontend
npm run build
```

### Environment Variables

EdukaAI Studio can be configured via environment variables. Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

**Key Configuration Options:**

```env
# Server Ports (change if ports are already in use)
EDUKAAI_PORT=8000                    # Backend API port (default: 8000)
VITE_PORT=3030                       # Frontend dev server port (default: 3030)

# API Connection (for custom backend locations)
VITE_API_URL=http://localhost:8000   # Backend API URL
VITE_WS_URL=ws://localhost:8000     # WebSocket URL

# Security
EDUKAAI_ALLOW_REMOTE=false          # Allow remote connections (not recommended)

# HuggingFace
EDUKAAI_HF_TOKEN=your-token         # For private models and higher rate limits

# Storage
EDUKAAI_STORAGE_PATH=./storage      # Data storage location
EDUKAAI_MAX_STORAGE_GB=50           # Max storage limit

# Database
EDUKAAI_DATABASE_URL=sqlite:///./storage/app/edukaai.db

# MLX / Training
EDUKAAI_MLX_DEVICE=gpu              # gpu or cpu
EDUKAAI_MAX_DATASET_SAMPLES=10000
EDUKAAI_MAX_CONTEXT_LENGTH=4096
```

See `.env.example` for all available options.

### Custom Port Configuration

If the default ports (8000 backend, 3030 frontend) are already in use:

**Option 1: Using .env file (Recommended)**
Edit `.env` in project root:
```env
EDUKAAI_PORT=8080
VITE_PORT=3000
VITE_API_URL=http://localhost:8080
VITE_WS_URL=ws://localhost:8080
```

Then start normally:
```bash
./launch.sh
```

**Option 2: Environment Variables**
```bash
# Terminal 1 - Backend on port 8080
export EDUKAAI_PORT=8080
cd backend && source .venv/bin/activate && python run.py

# Terminal 2 - Frontend on port 3000, connecting to backend on 8080
export VITE_PORT=3000
export VITE_API_URL=http://localhost:8080
export VITE_WS_URL=ws://localhost:8080
cd frontend && npm run dev
```


## Contributing

We welcome contributions! Contributing guide coming soon. 

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) - Apple's machine learning framework
- [HuggingFace](https://huggingface.co/) - Model hub and ecosystem
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [Vue.js](https://vuejs.org/) - Progressive JavaScript framework


****[AI Curator](https://github.com/elgap/ai-curator)** and **[EdukaAi Studio](https://github.com/elgap/edukaai-studio)** are part of EdukaAI project by Elgap** — making AI & fine-tuning accessible through open-source, no-code, zero config tools.

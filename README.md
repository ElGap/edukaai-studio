# EdukaAI Studio

> Fine-tune Large Language Models on Apple Silicon with an intuitive web interface

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![macOS](https://img.shields.io/badge/macOS-Apple%20Silicon-blue.svg)](https://www.apple.com/macos/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MLX](https://img.shields.io/badge/MLX-Apple%20Silicon-orange.svg)](https://github.com/ml-explore/mlx)

EdukaAI Studio is a user-friendly application for fine-tuning Large Language Models (LLMs) on your Apple Silicon Mac. Built on Apple's [MLX framework](https://github.com/ml-explore/mlx), it provides a simple 5-step wizard to train custom AI models - no coding required!

**Perfect for:**
- 🎓 Students learning about AI/ML
- 🏢 Professionals customizing models for specific domains
- 🔬 Researchers experimenting with fine-tuning
- 💡 Hobbyists exploring LLM capabilities

![EdukaAI Studio Overview](/EdukaAi-studio.png)

## 🚀 Quick Start (5 Minutes)

### One-Line Installation

**Option A: Copy & paste in Terminal**
```bash
curl -fsSL https://raw.githubusercontent.com/elgap/edukaai-studio/main/one-line-install.sh | bash
```

**Option B: Download and run**
```bash
# Download the installer
curl -fsSL -o install-edukai.sh https://raw.githubusercontent.com/elgap/edukaai-studio/main/one-line-install.sh

# Run it
bash install-edukai.sh
```

That's it! The installer will:
- ✅ Check your Mac compatibility (Apple Silicon required)
- ✅ Download EdukaAI Studio automatically
- ✅ Install Python dependencies
- ✅ Set up everything in `~/Applications/EdukaAI-Studio/`
- ✅ Create a Desktop shortcut

### Launch the Application

After installation completes:
```bash
# Option 1: Double-click 'EdukaAI Studio' on your Desktop
# Option 2: Run from Terminal:
~/Applications/EdukaAI-Studio/launch.sh
```

Then open your browser: **http://localhost:5173**

---

## 📖 Your First Fine-Tuning (10 Minutes)

### 1. Prepare Your Data

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

Alternatively, you can generate training samples interactively using [EdukaAI Dataset Manager](https://github.com/elgap/edukaai-dataset-manager), a locally-hosted tool with 100% privacy that provides a chat-like interface for creating structured training data—then simply export for EdukaAI Studio and fine-tune in minutes.

Save as `my-training-data.jsonl` (100-1000 examples recommended)

### Step 2: Upload & Train

1. Open EdukaAI Studio in your browser
2. Click "Upload Dataset" and select your file
3. Choose a base model:
   - **Tiny (500M-1B params)** - Fast training, basic tasks
   - **Small (1B-3B params)** - Good balance of speed & quality
   - **Medium (3B-7B params)** - Higher quality, requires more RAM
4. Select training preset:
   - **Quick** - 100 steps (~5 min) - Good for testing
   - **Balanced** - 300 steps (~15 min) - Recommended for most uses
   - **Thorough** - 1000 steps (~1 hour) - Maximum quality
5. Click "Start Training" and watch the magic happen!

### Step 3: Test Your Model

After training completes:
- Go to the **Dual Chat** page
- Compare responses from the base model vs. your fine-tuned version
- Export your model as:
  - **LoRA Adapter** (~10-50MB) - Use with base model
  - **Fused Model** - Standalone complete model
  - **GGUF** - For llama.cpp and local inference

---

## 🎯 Example Use Cases

### 1. Customer Support Bot
Train on past support tickets and responses to create a bot that understands your product.

### 2. Medical Q&A
Fine-tune on medical literature for a healthcare assistant (educational use only).

### 3. Code Assistant
Train on your codebase to get an AI that understands your specific coding patterns.

### 4. Creative Writing
Fine-tune on your favorite author's works to generate similar writing style.

### 5. Language Learning
Create a tutor for a specific language by training on dialogues and lessons.

---

## 🔧 Troubleshooting

### "Installation Failed"
- Make sure you have Python 3.10+ installed
- Check that you have at least 2GB free disk space
- Ensure macOS 12.0 or later

### "Out of Memory" during training
- Use a smaller base model (1B instead of 7B)
- Reduce batch size in advanced settings
- Close other applications to free up RAM

### "Training is slow"
- This is normal! Fine-tuning takes time
- Quick preset: ~5 minutes
- Balanced preset: ~15-30 minutes
- Thorough preset: ~1-2 hours

### Can't connect to localhost:5173
- Make sure both backend and frontend are running
- Try: `~/Applications/EdukaAI-Studio/launch.sh`
- Check if port 5173 is already in use by another app

---

## 🆘 Getting Help

**Documentation:** [Full docs in repository](https://github.com/elgap/edukaai-studio#readme)

**Report Issues:** [GitHub Issues](https://github.com/elgap/edukaai-studio/issues)

**Discussions:** [GitHub Discussions](https://github.com/elgap/edukaai-studio/discussions)

---

## 🏗️ For Developers

Select your base model from our curated list or add a custom MLX-compatible model from HuggingFace. Choose a training preset:

- **Quick** - 100 steps, fast iteration
- **Balanced** - 300 steps, good results
- **Thorough** - 1000 steps, maximum quality

Customize hyperparameters:
- LoRA Rank & Alpha
- Learning Rate
- Batch Size
- Training Steps
- Validation Split

### 3. Monitor Training

Watch real-time updates via WebSocket:
- Current step and loss
- Learning rate adjustments
- Best checkpoint tracking
- Resource utilization
- Live training logs

### 4. Export & Test

After training completes:
- Download LoRA adapters
- Export fused model
- Convert to GGUF format
- Test in side-by-side chat

## 🏗️ Architecture

```
EdukaAI Studio
├── Backend (FastAPI + MLX)
│   ├── REST API for configuration
│   ├── WebSocket for real-time updates
│   ├── Training Manager (async subprocess)
│   └── SQLite database for metadata
│
├── Frontend (Vue.js 3 + Tailwind)
│   ├── 5-step wizard interface
│   ├── Real-time log streaming
│   ├── Interactive charts
│   └── Responsive design
│
└── ML Pipeline (MLX + LoRA)
    ├── Model downloading & caching
    ├── LoRA fine-tuning
    ├── Checkpoint management
    └── Export to multiple formats
```

### Tech Stack

**Backend:**
- FastAPI - High-performance async web framework
- MLX - Apple's machine learning framework
- SQLAlchemy - Database ORM
- Pydantic - Data validation
- Python-multipart - File uploads

**Frontend:**
- Vue.js 3 - Progressive JavaScript framework
- Tailwind CSS - Utility-first CSS
- Pinia - State management
- Axios - HTTP client
- Recharts - Data visualization

**ML Libraries:**
- mlx-lm - Language model fine-tuning
- transformers - Model utilities
- safetensors - Efficient model storage

## 🛠️ Development

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
EDUKAAI_SECRET_KEY=your-secret       # Change in production!
EDUKAAI_ALLOW_REMOTE=false          # Allow remote connections (not recommended)

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

**Option 1: Environment Variables**
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

**Option 2: Using .env file**
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

## 📦 Packaging & Distribution

### Creating Distribution Package

Build a clean package for sharing:

```bash
# Create distributable .tar.gz
./build.sh 1.0.0

# Package will be created at:
# build/edukai-studio-1.0.0.tar.gz
```

### Creating macOS App Bundle

Create a clickable macOS app:

```bash
# Create .app bundle
./create-app.sh 1.0.0

# Install by dragging "EdukaAI Studio.app" to Applications folder
```

### Distribution Methods

**GitHub Releases:**
1. Create a new release on GitHub
2. Attach `edukai-studio-VERSION.tar.gz`
3. Add release notes

**Direct Download:**
- Share the .tar.gz file
- Users extract and run `./install.sh && ./launch.sh`

**macOS App:**
- Share the .app bundle or compressed .zip
- Users drag to Applications folder

## 🤝 Contributing

We welcome contributions! Contributing guide will be added soon.


## 📸 Screenshots

![Dataset Upload](docs/screenshots/datasets.png)
*Upload and validate training datasets with automatic format detection*

![Configure Training](docs/screenshots/configure.png)
*Select models, presets, and customize training parameters*

![Training Monitor](docs/screenshots/training.png)
*Real-time training logs and loss curves via WebSocket*

![Summary](docs/screenshots/summary.png)
*Detailed results with checkpoint management*

![Dual Chat](docs/screenshots/chat.png)
*Compare base vs fine-tuned models side-by-side*

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) - Apple's machine learning framework
- [HuggingFace](https://huggingface.co/) - Model hub and ecosystem
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [Vue.js](https://vuejs.org/) - Progressive JavaScript framework

## 📬 Contact

- **Issues**: [GitHub Issues](https://github.com/elgap/edukaai-studio/issues)
- **Discussions**: [GitHub Discussions](https://github.com/elgap/edukaai-studio/discussions)
- **Email**: Your email here (optional)
---



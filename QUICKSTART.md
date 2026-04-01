# EdukaAI Studio - Quick Start Guide

Visual step-by-step guide for beginners. Power users: check the 💡 Pro Tips in each section for advanced options.

---

## 🚀 Installation

### Step 1: Install (One Command)

```bash
mkdir edukaai-studio
cd edukaai-studio
curl -fsSL https://raw.githubusercontent.com/elgap/edukaai-studio/main/install.sh | bash
```

**What happens:**
- Downloads EdukaAI Studio
- Installs Python dependencies
- Sets up everything automatically

💡 **Pro Tip:** Add `--yes` flag for non-interactive install: `| bash -s -- --yes`

---

### Step 2: Launch

Run from terminal:
```bash
./launch.sh
```

**Open browser:** http://localhost:3030

💡 **Pro Tip:** The app runs entirely locally on your Mac. No data leaves your computer.

---

## 📊 Step-by-Step Workflow

### Step 1: Upload Your Dataset

![Step 1: Upload Dataset](EdukaAi-Studio.png)

**What you see:**
- Dataset upload area (drag & drop or click to browse)
- Optional validation dataset upload
- Auto-split validation option (splits your data into training/validation sets)

**Beginner:**
1. Prepare a `.jsonl` file with your training examples
2. Drag & drop the file into the upload area
3. Click "Configure Training →"

💡 **Pro Tips:**
- **100-1000 examples** recommended for good results
- **Quality > Quantity**: Clean, diverse examples work better than lots of repetitive data
- **Formats supported:** Alpaca (recommended), ShareGPT, simple Q&A

---

### Step 2: Configure Training

![Step 2: Configure Training](EdukaAi-Studio-configure.png)

**What you see:**
- Base model selection table
- Training preset options
- Advanced configuration (expandable)
- Summary panel on the right
- "Start Training" button

**Beginner - Quick Start:**
1. **Select Base Model**: Start with small models (1B-3B parameters)
   - Qwen 2.5 0.5B or 1.5B for testing
   - Llama 3.2 1B for English tasks
2. **Choose Preset**:
   - **Quick** (100 steps): ~5 min - For testing your setup
   - **Balanced** (300 steps): ~15 min - **Recommended for most users**
   - **Thorough** (1000 steps): ~1 hour - Best quality, longer wait
3. Click **"Start Training →"**

💡 **Pro Tips:**
- **Model Size Guide:**
  - 8GB RAM: Use models under 1B parameters
  - 16GB RAM: Can handle up to 3B parameters  
  - 32GB+ RAM: Can train 7B+ parameter models
- **Custom Models:** Click "+ Add Custom Model" to use any HuggingFace model
- **Advanced Settings:** Expand to customize learning rate, LoRA rank, batch size
- **PII Detection:** Check "Enable PII Detection" (experimental) to anonymize personal data

---

### Step 3: Monitor Training

![Step 3: Training Progress](EdukaAi-Studio-training.png)

**What you see:**
- Real-time loss curve
- Training metrics (loss, learning rate, speed)
- Progress bar and estimated time remaining
- Current step and total steps

**Beginner:**
- Watch the **Loss** go down - this is good!
- Loss should decrease steadily (not jump around wildly)
- Wait for training to complete (checkpoints save automatically)

**Understanding the Metrics:**
- **Loss**: How wrong the model is (lower = better). Should start high and decrease.
- **Learning Rate**: How fast the model learns (adjusts automatically)
- **Speed**: Tokens processed per second
- **ETA**: Estimated time to completion

💡 **Pro Tips:**
- **Early Stopping**: Training stops automatically if loss stops improving
- **Checkpoints**: Best checkpoint is auto-selected, but all are saved
- **Pause/Resume**: You can pause training and resume later
- **Performance**: Training is faster when your Mac is plugged in

---

### Step 4: Review Results

![Step 4: Training Summary](EdukaAi-Studio-summary.png)

**What you see:**
- Training summary statistics
- Final loss and metrics
- Dataset information
- Export options

**Beginner:**
- Review the final loss (lower is better)
- Check that training completed successfully
- Click "Dual Chat" to test your model
- Or click "Export Model" to save it

💡 **Pro Tips:**
- **Loss Interpretation:** 
  - < 1.0: Excellent
  - 1.0-2.0: Good
  - > 2.0: May need more training or better data
- **Export Options:**
  - **LoRA Adapter** (~10-50MB): Small, works with base model
  - **Fused Model** (1-7GB): Complete standalone model

---

### Step 5: Test & Compare (Dual Chat)

![Step 5: Dual Chat](EdukaAi-Studio-dual-chat.png)

**What you see:**
- Side-by-side model comparison
- Original base model on the left
- Your fine-tuned model on the right
- Chat interface for both models

**Beginner:**
- Type a question in either chat box
- Compare responses side-by-side
- See how your fine-tuned model differs from the base
- Test with questions related to your training data

**Example Test Questions:**
- Ask about topics from your training data
- Compare tone and style differences
- Check if your model learned the patterns you taught it

💡 **Pro Tips:**
- **System Prompts:** Customize how your model behaves
- **Temperature:** Lower = more focused, Higher = more creative
- **Max Tokens:** Control response length
- **Quick Compare:** Great for A/B testing different training runs

---

### Step 6: Manage Your Models (My Models)

![Step 6: My Models](EdukaAiStudio-my-models.png)

**What you see:**
- List of all your trained models
- Training history and metadata
- Quick actions (Export, Chat, Delete)
- Model details (base model, dataset, training steps)

**Beginner:**
- View all your past training runs in one place
- Click on any model to see details
- Export models you've trained
- Delete old models to free up space

**Navigation:**
- Click "My Models" in the top navigation bar
- Or visit: http://localhost:3030/models

💡 **Pro Tips:**
- **Organize by name**: Use descriptive names like "customer-support-v1" or "code-assistant-python"
- **Compare runs**: See which training configuration worked best
- **Storage management**: Delete old or failed runs to free disk space
- **Re-export**: Export the same model in different formats (LoRA, Fused, GGUF)
- **Quick test**: Click "Chat" to quickly test any model without going through Dual Chat

---

## 🎯 Example Use Cases

### Customer Support Bot
```jsonl
{"instruction": "How do I reset my password?", "input": "", "output": "Go to Settings > Security > Reset Password. Enter your email and click 'Send Reset Link'. Check your inbox for the email and follow the link."}
{"instruction": "What are your business hours?", "input": "", "output": "We're open Monday-Friday 9AM-6PM EST. Live chat support is available during these hours."}
```

### Code Assistant
```jsonl
{"instruction": "Explain this Python code", "input": "def fib(n): return n if n <= 1 else fib(n-1)+fib(n-2)", "output": "This is a recursive Fibonacci sequence generator. It calculates the nth Fibonacci number by recursively calling itself for the two preceding numbers."}
{"instruction": "Write a function to reverse a string", "input": "", "output": "```python\ndef reverse_string(s):\n    return s[::-1]\n```"}
```

### Creative Writing
```jsonl
{"instruction": "Write a haiku about autumn", "input": "", "output": "Golden leaves descend\nCrisp air whispers through bare trees\nAutumn's quiet song"}
```

---

## 🆘 Troubleshooting

### "Out of Memory"
**Solution:** Use smaller model or reduce batch size in advanced settings
- 8GB RAM: Stick to models under 1B parameters
- Try reducing batch size from 4 to 1 or 2

### "Training is very slow"
**Normal!** Training takes time. Speed up:
- Plug in your Mac (performance mode)
- Close other applications
- Use Quick preset (100 steps) for testing
- Smaller models train faster

### "Can't connect to localhost:3030"
**Solution:**
```bash
./launch.sh
```
Check that both backend and frontend are running.

### "Installation failed"
**Check:**
- Python 3.10+ installed? (`python3 --version`)
- macOS 12.0+? (`sw_vers`)
- 2GB+ free space? (`df -h`)
- Internet connection working?

---

## 📚 Supported File Formats

### Input Formats

**1. Alpaca Format (Recommended)**
```jsonl
{"instruction": "Summarize this", "input": "Long text here...", "output": "Short summary"}
```

**2. ShareGPT/Chat Format**
```jsonl
{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
```

**3. Simple Text Completion**
```jsonl
{"text": "Once upon a time in a land far away..."}
```

### Output Formats Explained

**LoRA Adapter (.safetensors)**
- Contains only the "learning" from training
- Must be used with the original base model
- Great for sharing and iterative improvements

**Fused Model (.safetensors)**
- Complete model including base + training
- Can be used standalone
- Larger file size but self-contained

---

## ⚡ Performance Tips

### RAM Management
| Your Mac | Max Model Size | Recommended |
|----------|---------------|-------------|
| 8GB RAM | < 1B params | Qwen 0.5B |
| 16GB RAM | Up to 3B params | Llama 3.2 1B-3B |
| 32GB+ RAM | 7B+ params | Mistral 7B, Llama 3 8B |

### Training Speed Guide
| Preset | Steps | Typical Time | Use For |
|--------|-------|--------------|---------|
| Quick | 100 | 5-15 min | Testing setup |
| Balanced | 300 | 15-45 min | **Most users** |
| Thorough | 1000 | 1-3 hours | Best quality |

*Times vary based on model size and Mac speed (M4 > M3 > M2 > M1)*

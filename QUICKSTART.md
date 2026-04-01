# EdukaAI Studio - Quick Reference Guide

## 🚀 Installation (One Command)

```bash
curl -fsSL https://raw.githubusercontent.com/elgap/edukaai-studio/main/install.sh | bash
```

**After installation:**
- Double-click "EdukaAI-Studio.command" on your Desktop
- Or run: `cd edukaai-studio && ./launch.sh`

Then open browser: **http://localhost:3030**

**Note:** App is installed in `./edukaai-studio/` (current directory)

---

## 📊 Quick Start Workflow

### 1. Prepare Data (5 min)
Create a `.jsonl` file with examples:

```jsonl
{"instruction": "What is photosynthesis?", "input": "", "output": "Photosynthesis is how plants convert sunlight into energy..."}
{"instruction": "Explain neural networks", "input": "", "output": "Neural networks are computing systems with interconnected nodes..."}
```

**Tips:**
- 100-1000 examples for good results
- More examples = better model
- Quality over quantity

### 2. Upload & Train (15-60 min)

1. **Upload Dataset** → Drag & drop your `.jsonl` file
2. **Select Model** → Start with small (1B-3B params)
3. **Choose Preset**:
   - Quick: 100 steps (~5 min) - For testing
   - Balanced: 300 steps (~15 min) - Recommended
   - Thorough: 1000 steps (~1 hour) - Best quality
4. **Start Training** → Watch the loss curve go down!

### 3. Export & Use (2 min)

After training:
- Click "Dual Chat" to compare models
- Export as:
  - **LoRA Adapter** - Small file, use with base model
  - **Fused Model** - Complete standalone model
  - **GGUF** - For local tools like Ollama

---

## 💡 Pro Tips

### RAM Management
- **8GB RAM:** Use models under 1B parameters
- **16GB RAM:** Can handle up to 3B parameters
- **32GB+ RAM:** Can train 7B+ parameter models

**If you run out of memory:**
- Use smaller base model
- Reduce batch size in advanced settings
- Close other applications

### Training Speed
Training time depends on:
- Model size (larger = slower)
- Number of training steps
- Your Mac's speed (M4 > M3 > M2 > M1)

Typical times:
- Quick (100 steps): 5-15 minutes
- Balanced (300 steps): 15-45 minutes
- Thorough (1000 steps): 1-3 hours

### Best Practices
1. **Start small** - Test with Quick preset first
2. **Quality data** - Clean, diverse examples work best
3. **Monitor loss** - Should go down steadily
4. **Save checkpoints** - Best checkpoint auto-selected
5. **Export both** - Get LoRA + Fused for flexibility

---

## 🆘 Common Issues

### "Out of Memory"
**Solution:** Use smaller model or reduce batch size

### "Training is very slow"
**Normal!** Training takes time. Check:
- Is your Mac plugged in? (Performance mode)
- Close other apps
- Use Quick preset for testing

### "Can't connect to localhost:3030"
**Solution:** 
```bash
cd edukaai-studio && ./launch.sh
```

### "Installation failed"
**Check:**
- Python 3.10+ installed?
- macOS 12.0+?
- 2GB+ free space?
- Internet connection working?

---

## 📚 File Formats

### Supported Input Formats

**1. Alpaca Format (Recommended)**
```jsonl
{"instruction": "Summarize this", "input": "Long text here...", "output": "Short summary"}
```

**2. Simple Q&A**
```
Question: What is X?
Answer: X is...
```

**3. Chat/Conversation**
```jsonl
{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
```

### Output Formats

**LoRA Adapter (.safetensors)**
- Size: ~10-50MB
- Use with: Base model
- Best for: Iterative improvements

**Fused Model (.safetensors)**
- Size: Same as base model (1-7GB)
- Use with: Direct inference
- Best for: Standalone deployment

**GGUF (.gguf)**
- Size: Varies by quantization
- Use with: llama.cpp, Ollama
- Best for: Local deployment tools

---

## 🎯 Example Datasets

### Customer Support
```jsonl
{"instruction": "How do I reset my password?", "input": "", "output": "Go to Settings > Security > Reset Password..."}
```

### Code Assistant
```jsonl
{"instruction": "Explain this Python code", "input": "def fib(n): return n if n <= 1 else fib(n-1)+fib(n-2)", "output": "This is a recursive Fibonacci sequence generator..."}
```

### Creative Writing
```jsonl
{"instruction": "Write a haiku about autumn", "input": "", "output": "Golden leaves descend\nCrisp air whispers through bare trees\nAutumn's quiet song"}
```

---

## 📞 Support

**Having trouble?**
1. Check README.md for detailed docs
2. Visit: https://github.com/elgap/edukaai-studio/issues
3. Join discussions: https://github.com/elgap/edukaai-studio/discussions

---

**Happy Fine-Tuning! 🚀**

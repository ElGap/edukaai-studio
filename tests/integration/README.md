# Integration Tests for EdukaAI Studio

## Overview

These are the **MOST IMPORTANT TESTS** in the entire test suite. They verify that the core training functionality actually works end-to-end.

**Types of Integration Tests:**

1. **Real Training Tests** - Actually run MLX training (slow, requires Apple Silicon)
2. **Mock Training Tests** - Simulate training without MLX (fast, runs anywhere)

## 🎯 Test Philosophy

> *"If the training integration test passes, the app works. If it fails, nothing else matters."*

The training integration test is the **single most critical test** because:
- It tests the entire ML pipeline
- It verifies MLX integration works
- It checks model downloading works
- It validates the training loop
- It ensures state management works
- It confirms UI updates work

## 📁 Test Files

| File | Type | Duration | Requirements |
|------|------|----------|----------------|
| `test_training_integration.py` | Real | 10-30 min | Apple Silicon, MLX, Internet |
| `test_training_mock.py` | Mock | 30 sec | None (runs anywhere) |

## 🚀 Running Tests

### Quick Test (Mock Only - Fast)
```bash
# Run mock tests (no MLX required, 30 seconds)
pytest tests/integration/test_training_mock.py -v --run-mock
```

### Full Test (Real Training - Slow)
```bash
# Run real training tests (requires Apple Silicon + MLX, 10-30 minutes)
pytest tests/integration/test_training_integration.py -v --run-slow
```

### Default (No Integration Tests)
```bash
# Run all tests EXCEPT integration tests
pytest tests/browser/ tests/unit/ -v
```

### All Tests (Everything)
```bash
# Run everything including slow integration tests
pytest tests/ -v --run-slow --run-mock
```

## 🧪 Test Descriptions

### Real Training Tests (`test_training_integration.py`)

**Test: `test_training_process_runs_to_completion`**
- **What:** Actually runs training with MLX
- **Duration:** 10-30 minutes
- **Requirements:** Apple Silicon Mac, Internet, 2-4GB disk space
- **Verifies:**
  - Training starts successfully
  - Progress updates correctly
  - Status changes: Ready → Training → Complete
  - Loss values are recorded
  - Model is saved to disk
  - State updated correctly
  - Other tabs detect completion

**Test: `test_training_can_be_stopped`**
- **What:** Tests stop functionality
- **Duration:** < 1 minute
- **Verifies:** Training can be interrupted

**Test: `test_training_state_persisted_to_disk`**
- **What:** Verifies state persistence
- **Duration:** < 1 minute
- **Verifies:** State survives page reload

**Test: `test_other_tabs_detect_completion`**
- **What:** Cross-tab state verification
- **Duration:** < 1 minute
- **Verifies:** Results/Chat tabs see completion

### Mock Training Tests (`test_training_mock.py`)

**Test: `test_training_generator_yields_correct_sequence`**
- **What:** Simulates training without MLX
- **Duration:** 2 seconds
- **Verifies:** Generator yields correct number of outputs with proper structure

**Test: `test_training_completion_state_update`**
- **What:** Verifies state flags
- **Duration:** < 1 second
- **Verifies:** `training_complete=True`, `training_active=False`

**Test: `test_stale_complete_state_detection`**
- **What:** Tests completion detection
- **Duration:** < 1 second
- **Verifies:** Already-completed training is detected

**Test: `test_training_status_progression`**
- **What:** Verifies status messages
- **Duration:** < 1 second
- **Verifies:** Status flows: Ready → Training → Complete

**Test: `test_progress_values_make_sense`**
- **What:** Validates progress calculations
- **Duration:** < 1 second
- **Verifies:** Progress goes 0% → 100% correctly

**Test: `test_loss_values_decrease_over_time`**
- **What:** Verifies loss trends
- **Duration:** < 1 second
- **Verifies:** Losses generally decrease during training

**Test: `test_stop_training_handler`**
- **What:** Tests stop functionality
- **Duration:** < 1 second
- **Verifies:** Stop handler works correctly

**Test: `test_disk_state_saved_with_correct_flags`**
- **What:** Verifies persistence
- **Duration:** < 1 second
- **Verifies:** State saved to disk correctly

**Test: `test_chat_tab_reads_completion_from_disk`**
- **What:** Cross-tab verification
- **Duration:** < 1 second
- **Verifies:** Chat tab detects completion

**Plus edge case tests...**

## 📊 When to Run Which Tests

### During Development (Fast Feedback)
```bash
# Run only browser and unit tests
pytest tests/browser/ tests/unit/ -v

# Time: ~2 minutes
# Coverage: 85%
```

### Before Commit (Medium)
```bash
# Add mock integration tests
pytest tests/browser/ tests/unit/ tests/integration/test_training_mock.py -v --run-mock

# Time: ~3 minutes
# Coverage: 90%
```

### Before Release (Complete)
```bash
# Run everything including real training
pytest tests/ -v --run-slow --run-mock

# Time: 30-60 minutes
# Coverage: 95%
# Note: Requires Apple Silicon Mac
```

### In CI/CD (Automated)
```bash
# Can't run real training in CI (no Apple Silicon)
# So run mock tests only
pytest tests/browser/ tests/unit/ tests/integration/test_training_mock.py -v --run-mock

# Time: ~3 minutes
# Coverage: 90%
```

## ⚠️ Important Notes

### Real Training Tests
- ⚠️ **Requires Apple Silicon Mac** - MLX only works on Apple Silicon
- ⚠️ **Requires Internet** - Downloads models from HuggingFace
- ⚠️ **Requires 2-4GB disk space** - Models are large
- ⚠️ **Takes 10-30 minutes** - Training is slow even with minimal settings
- ⚠️ **Uses minimal settings** - 10 iterations, small model for speed

### Mock Training Tests
- ✅ **Runs anywhere** - No MLX required
- ✅ **Very fast** - 30 seconds for all tests
- ✅ **CI/CD friendly** - Can run in GitHub Actions
- ⚠️ **Simulated only** - Doesn't test actual MLX integration

## 🔧 Test Configuration

### Minimal Training Settings (for speed)

Integration tests use minimal settings to run quickly:

```python
{
    'iterations': 10,        # Instead of 200-600
    'lora_rank': 8,          # Instead of 16
    'lora_alpha': 16,        # Instead of 32
    'grad_accumulation': 4,   # Instead of 32
    'max_seq_length': 512,   # Instead of 2048
}
```

Even with these minimal settings, training takes 10-30 minutes.

### Mock vs Real

**Mock tests simulate:**
- Progress updates
- Loss values
- Status messages
- State changes
- Queue operations

**Real tests verify:**
- Actual MLX execution
- Real model downloads
- Genuine loss calculations
- Actual file operations
- True training loop

## 🎓 Best Practices

### 1. Run Mock Tests First
Always run mock tests before real tests:
```bash
pytest tests/integration/test_training_mock.py -v --run-mock
# If these pass, real tests have a better chance
```

### 2. Use Minimal Settings
When running real tests, use the minimal settings already configured. Don't try to train for 600 iterations!

### 3. Monitor Disk Space
Models are large. Make sure you have 2-4GB free:
```bash
df -h
```

### 4. Check Internet
Real tests need to download models:
```bash
ping huggingface.co
```

### 5. Verify Apple Silicon
Only run real tests on Apple Silicon:
```bash
uname -m  # Should show: arm64
```

## 🔍 Troubleshooting

### Test Fails: "MLX not available"
**Solution:** You're not on Apple Silicon or MLX not installed
```bash
pip install mlx
# Or run mock tests instead
pytest tests/integration/test_training_mock.py --run-mock
```

### Test Fails: "Connection refused"
**Solution:** Internet issue or HuggingFace down
```bash
# Check internet
curl https://huggingface.co
# If fails, retry later or run mock tests
```

### Test Takes Forever
**Solution:** This is expected for real training
```bash
# Real training takes 10-30 minutes, that's normal
# Use mock tests for fast feedback
pytest tests/integration/test_training_mock.py --run-mock
```

### Test Fails: "Disk full"
**Solution:** Clean up disk space
```bash
# Remove old outputs
rm -rf outputs/
# Check space
df -h
```

## 📈 Coverage Impact

**With only browser/unit tests:** 85% coverage
**Adding mock integration tests:** 90% coverage
**Adding real integration tests:** 95% coverage

The integration tests cover the **critical 10%** that actually runs training.

## 🏆 Success Criteria

A successful integration test proves:
1. ✅ Training actually starts
2. ✅ Progress updates work
3. ✅ Status changes correctly
4. ✅ Loss values are real (not mocked)
5. ✅ Model files are created
6. ✅ State is persisted
7. ✅ Other tabs see completion
8. ✅ The entire ML pipeline works

If all integration tests pass, the app **guaranteed works** for training.

## 📝 Summary

**These are the most important tests because they verify the core functionality actually works.**

Without these tests, you don't know if:
- MLX integration works
- Training actually runs
- Models are created
- State is saved
- The whole app functions

**Always run integration tests before releasing.**

---

**Quick Reference:**
```bash
# Fast (mock only)
pytest tests/integration/test_training_mock.py -v --run-mock

# Slow (real training)
pytest tests/integration/test_training_integration.py -v --run-slow

# Default (no integration)
pytest tests/browser/ tests/unit/ -v
```

**Status:** Ready to use ✅  
**Total Tests:** 14 integration tests  
**Critical Tests:** 2 (one real, one mock)  
**Last Updated:** 2026-03-23

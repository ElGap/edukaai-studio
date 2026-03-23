# EdukaAI Studio Test Suite

## Overview

This test suite catches **real bugs** while avoiding **false positives**. It only flags issues that will cause actual runtime errors, not stylistic preferences or informational patterns.

## What Tests Catch

### 🔴 Critical Issues (Must Fix)

1. **Deprecated Gradio API Calls**
   - Pattern: `gr.Dropdown.update()`, `gr.Textbox.update()`, etc.
   - Error: `AttributeError: type object 'Dropdown' has no attribute 'update'`
   - Fix: Use `gr.update()` instead
   - **This caused the crash you experienced!**

2. **Mismatched Yield Outputs**
   - Pattern: Generator functions with yields returning different numbers of values
   - Error: Gradio expects consistent outputs from all yields in a function
   - Fix: Ensure all yields return same number of elements
   - **This causes "Function didn't return enough output values" errors**

3. **Broken Yield Patterns**
   - Pattern: Syntax errors in yield statements (empty yields, mismatched brackets)
   - Error: `SyntaxError` at runtime
   - Fix: Properly format yield statements

### ⚠️ Warnings (Should Review)

4. **Missing Component References**
   - Pattern: Using `components['key']` before it's defined
   - Error: `KeyError` at runtime
   - Fix: Define components before referencing them

5. **Bad Function Signatures**
   - Pattern: Docstring says returns Tuple but returns single value
   - May cause issues with Gradio's output validation

## What Tests DON'T Catch (False Positives Avoided)

❌ `current_state.get()` calls - These are legitimate, just reading state  
❌ Function complexity - Not a bug  
❌ Code style - Not relevant to functionality  
❌ Import ordering - Not relevant to functionality  

## Running Tests

### Quick Test (All tabs)

```bash
cd /Users/developer/Projects/edukaai-studio
python3 tests/test_realistic.py
```

### With Verbose Output

```bash
python3 tests/test_realistic.py --verbose
```

### Test Specific File

```bash
# Currently only tests files in ui/tabs/ directory
python3 tests/test_realistic.py
```

## Test Output Examples

### ✅ All Tests Pass

```
Testing: chat.py
  ✅ No critical issues found
Testing: configure.py
  ✅ No critical issues found
...
======================================================================
✅ All tests passed! No critical issues found.
======================================================================
```

### ❌ Issues Found

```
Testing: chat.py
  ❌ Found 2 critical, 1 warning(s):
    🔴 Line 781: Replace gr.Dropdown.update(...) with gr.update(...)
       Code: return gr.Dropdown.update(choices=choices, ...
    🔴 Line 801: Replace gr.Dropdown.update(...) with gr.update(...)
       Code: return gr.Dropdown.update(choices=choices, ...
    ⚠️  Line 245: Component 'components['missing_key']' used before definition
```

## CI/CD Integration

Add to your CI pipeline:

```yaml
# .github/workflows/test.yml
- name: Run UI Tests
  run: python3 tests/test_realistic.py
```

Or in a pre-commit hook:

```bash
#!/bin/bash
# .git/hooks/pre-commit
python3 tests/test_realistic.py || exit 1
```

## Common Fixes

### Fix: gr.Component.update() → gr.update()

**Before:**
```python
return gr.Dropdown.update(choices=my_choices, value=my_value)
```

**After:**
```python
return gr.update(choices=my_choices, value=my_value)
```

### Fix: Mismatched Yield Counts

**Before:**
```python
def my_generator():
    yield [1, 2, 3]      # 3 outputs
    yield [1, 2]         # 2 outputs ❌
    yield [1, 2, 3, 4]   # 4 outputs ❌
```

**After:**
```python
def my_generator():
    yield [1, 2, 3, ""]  # 4 outputs with placeholder
    yield [1, 2, "", ""]  # 4 outputs with placeholders
    yield [1, 2, 3, 4]    # 4 outputs ✓
```

## Architecture

The test suite uses:
- **Regular expressions** for fast pattern matching on deprecated APIs
- **AST parsing** for accurate yield count analysis
- **Line-by-line analysis** for component reference tracking

## Why Not Use Standard Linters?

Standard Python linters (flake8, pylint, mypy) don't understand Gradio-specific patterns. This suite:
- Knows Gradio 6.x deprecated `gr.Component.update()`
- Understands Gradio's requirement for consistent yield outputs
- Tracks component definition order specific to Gradio patterns

## Troubleshooting

**Tests pass but I still get errors:**
- Tests only catch patterns we know about. You may have found a new bug pattern!
- Add the pattern to `tests/test_realistic.py` to catch it in the future

**False positives:**
- If a test flags something that isn't actually a bug, please report it
- The goal is zero false positives

**Missing tests:**
- The suite only tests files in `src/edukaai_studio/ui/tabs/`
- Add new test functions for new bug patterns

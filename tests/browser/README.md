# Browser Testing with Playwright

## Overview

This directory contains browser-based integration tests using Playwright. These tests verify that the UI works correctly in a real browser, complementing the unit tests.

## Current Test Coverage

### ✅ Implemented Tests

1. **test_upload.py** - Upload tab functionality
   - Page loads correctly
   - Valid JSONL upload works
   - Invalid files show errors
   - Navigation button provides helpful messages
   - Preview modes work

2. **test_configure.py** - Configure tab functionality
   - Page loads correctly
   - Model dropdown populated
   - Preset selection works
   - Parameters visible

3. **test_chat.py** - Chat tab functionality (NEW BASE-ONLY MODE)
   - Page loads correctly
   - Base model selector visible
   - Can select "Use Base Model Only"
   - Chat without training works

4. **test_workflow.py** - End-to-end workflows
   - Upload → Configure flow
   - Base model chat workflow
   - All tabs navigable
   - Workflow guides visible
   - Error handling graceful

## Installation

### Step 1: Install Playwright

```bash
# Install Playwright for Python
pip install pytest-playwright

# Install browser binaries (only need Chromium for now)
playwright install chromium

# Or install all browsers
playwright install
```

### Step 2: Verify Installation

```bash
# Check Playwright is installed
python -c "import playwright; print(playwright.__version__)"

# Check browsers installed
playwright install --help
```

## Running Tests

### Quick Test (Headless - for CI)

```bash
# Start the app in background
python src/edukaai_studio/main_simplified.py &

# Wait for app to start (10 seconds)
sleep 10

# Run all browser tests
pytest tests/browser/ --headed=false -v

# Run specific test file
pytest tests/browser/test_upload.py --headed=false -v

# Run specific test
pytest tests/browser/test_upload.py::test_upload_valid_jsonl --headed=false -v
```

### Visual Test (Headed - for debugging)

```bash
# Run with browser visible (great for debugging)
pytest tests/browser/test_upload.py --headed -v --slowmo=1000

# The --slowmo=1000 adds 1 second delay between actions so you can see what's happening
```

### With Screenshots on Failure

```bash
# Capture screenshots when tests fail
pytest tests/browser/ --headed=false --screenshot=only-on-failure -v

# Screenshots saved to test-results/
```

### With Video Recording

```bash
# Record video of test execution
pytest tests/browser/ --headed=false --video=on -v

# Videos saved to test-results/
```

## Test Structure

```
tests/browser/
├── conftest.py           # Shared fixtures and configuration
├── test_upload.py        # Upload tab tests
├── test_configure.py     # Configure tab tests
├── test_chat.py          # Chat tab tests
├── test_workflow.py      # End-to-end workflow tests
└── __init__.py
```

## Fixtures (conftest.py)

### `gradio_app` (session scope)
Starts the Gradio app for testing.

```python
def test_something(page: Page, gradio_app):
    # App is already running at localhost:7860
    page.goto("http://localhost:7860")
```

### `page` (function scope)
Provides a fresh browser page for each test.

```python
def test_something(page: Page):
    page.goto("http://localhost:7860")
    # Test with this page
```

### `sample_data_path`
Path to sample training data.

```python
def test_upload(page: Page, sample_data_path):
    page.set_input_files(str(sample_data_path))
```

## Writing New Tests

### Basic Test Structure

```python
from playwright.sync_api import Page, expect

def test_descriptive_name(page: Page):
    """Test description."""
    # 1. Navigate
    page.goto("http://localhost:7860")
    
    # 2. Interact
    page.get_by_role("button", name="Click Me").click()
    
    # 3. Verify
    expect(page.get_by_text("Expected Result")).to_be_visible()
```

### Common Patterns

#### Clicking Tabs
```python
page.get_by_role("tab", name="1. Upload").click()
```

#### Setting File Input
```python
page.get_by_label("Training Data").set_input_files("/path/to/file.jsonl")
```

#### Selecting Dropdown
```python
page.get_by_label("Model").select_option("phi-3-mini")
```

#### Waiting for Text
```python
# Wait up to 10 seconds
expect(page.get_by_text("Success!")).to_be_visible(timeout=10000)
```

#### Checking Attribute
```python
expect(page.get_by_role("tab")).to_have_attribute("aria-selected", "true")
```

## Debugging Failed Tests

### Method 1: Visual Debugging
```bash
# Run with browser visible and slow motion
pytest tests/browser/test_upload.py::test_name --headed --slowmo=2000 -v
```

### Method 2: Screenshots
```bash
# Screenshot on failure
pytest tests/browser/ --screenshot=only-on-failure

# Then check test-results/ directory
ls test-results/
```

### Method 3: Trace Viewer
```bash
# Record trace (includes screenshots, console logs, network)
pytest tests/browser/ --tracing=on

# Open trace viewer
playwright show-trace test-results/trace.zip
```

### Method 4: Debug Mode
```python
# Add to your test
def test_something(page: Page):
    page.goto("http://localhost:7860")
    
    # Pause here to inspect manually
    page.pause()
    
    # Continue after manual inspection
    page.get_by_role("button").click()
```

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/browser-tests.yml
name: Browser Tests
on: [push, pull_request]

jobs:
  browser-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest-playwright
      
      - name: Install Playwright browsers
        run: playwright install chromium
      
      - name: Start app and run tests
        run: |
          python src/edukaai_studio/main_simplified.py &
          sleep 10
          pytest tests/browser/ --headed=false -v --screenshot=only-on-failure
      
      - name: Upload screenshots on failure
        if: failure()
        uses: actions/upload-artifact@v3
        with:
          name: test-screenshots
          path: test-results/
```

## Troubleshooting

### App Won't Start
```bash
# Check if port is in use
lsof -i :7860

# Kill existing process
kill $(lsof -t -i:7860)
```

### Tests Fail with Timeout
```bash
# App might need more time to start
sleep 15  # Increase from 10 to 15 seconds

# Or increase test timeout
pytest tests/browser/ --timeout=120
```

### Browser Not Found
```bash
# Reinstall browsers
playwright install chromium

# Or check installation
playwright install --help
```

### Element Not Found
```bash
# Run in headed mode to see what's happening
pytest tests/browser/test_name.py --headed --slowmo=2000

# Check if selectors need updating
# Gradio changes class names between versions
```

## Best Practices

1. **Use data-testid attributes** when possible
   ```python
   # In Gradio: elem_id="upload-button"
   page.get_by_test_id("upload-button").click()
   ```

2. **Wait for network idle** after async operations
   ```python
   page.wait_for_load_state("networkidle")
   ```

3. **Take screenshots before assertions** when debugging
   ```python
   page.screenshot(path="debug.png")
   ```

4. **Use semantic selectors** (get_by_role, get_by_label) instead of CSS
   ```python
   # Good
   page.get_by_role("button", name="Submit")
   
   # Avoid
   page.locator(".button.primary")
   ```

5. **Clean up in finally blocks** for temp files
   ```python
   try:
       # Test code
   finally:
       os.unlink(temp_path)
   ```

## Coverage Goals

### Current (Manual Testing Only)
- ❌ No automated browser tests
- ❌ Hard to catch UX regressions

### Target (With These Tests)
- ✅ Upload tab fully tested
- ✅ Configure tab fully tested
- ✅ Chat tab fully tested
- ✅ Critical workflows tested
- ✅ Can run in CI/CD

### Future (Next Phase)
- ⏳ Train tab tests (requires long-running tests)
- ⏳ Results tab download tests
- ⏳ My Models tab tests
- ⏳ Cross-browser testing (Firefox, Safari)
- ⏳ Mobile viewport testing

## Resources

- [Playwright Documentation](https://playwright.dev/python/)
- [Playwright Best Practices](https://playwright.dev/python/docs/best-practices)
- [pytest-playwright](https://github.com/microsoft/playwright-pytest)
- [Gradio Testing](https://www.gradio.app/guides/testing-your-app)

## Questions?

If tests are failing:
1. Run with `--headed` to see what's happening
2. Check `test-results/` for screenshots
3. Verify app is running at localhost:7860
4. Check that Gradio version matches what tests expect

---

**Last Updated:** 2026-03-23  
**Playwright Version:** Latest  
**Browsers:** Chromium  
**Status:** Ready to use 🚀

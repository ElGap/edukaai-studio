"""Playwright configuration and fixtures for EdukaAI Studio browser tests."""

import pytest
from playwright.sync_api import Page, Browser, BrowserContext
from pathlib import Path
import subprocess
import time
import signal


@pytest.fixture(scope="session")
def gradio_app():
    """Start the Gradio app for testing."""
    # Start the app
    process = subprocess.Popen(
        ["python", "src/edukaai_studio/main_simplified.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=Path(__file__).parent.parent.parent
    )
    
    # Wait for app to start
    time.sleep(10)
    
    yield process
    
    # Cleanup
    process.send_signal(signal.SIGTERM)
    process.wait(timeout=5)


@pytest.fixture
def page(browser: Browser, gradio_app) -> Page:
    """Create a new browser page for each test."""
    context: BrowserContext = browser.new_context(
        viewport={"width": 1280, "height": 720}
    )
    page: Page = context.new_page()
    
    # Navigate to app
    page.goto("http://localhost:7860")
    
    # Wait for app to load
    page.wait_for_selector("text=EdukaAI", timeout=10000)
    
    yield page
    
    context.close()


@pytest.fixture
def sample_data_path() -> Path:
    """Path to sample training data."""
    return Path(__file__).parent.parent / "data" / "sample.jsonl"

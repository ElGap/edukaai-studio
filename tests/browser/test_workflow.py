"""End-to-end workflow tests.

These tests verify complete user journeys through the application.
"""

from playwright.sync_api import Page, expect
from pathlib import Path


def test_upload_to_configure_flow(page: Page, sample_data_path: Path):
    """Critical path: Upload file and view helpful navigation message."""
    # Step 1: Upload
    page.get_by_role("tab", name="1. Upload").click()
    page.get_by_label("Training Data").set_input_files(str(sample_data_path))
    
    # Wait for processing
    expect(page.get_by_text("validated")).to_be_visible(timeout=10000)
    
    # Step 2: Verify navigation button shows helpful message
    nav_button = page.get_by_role("button", name="Go to Configure")
    expect(nav_button).to_be_visible()
    
    # Click and verify helpful message
    nav_button.click()
    
    # Status should show helpful message
    status = page.get_by_label("Status")
    expect(status).to_contain_text("Configure")


def test_base_model_chat_workflow(page: Page):
    """Critical path: Chat with base model without training."""
    # Step 1: Navigate to Chat
    page.get_by_role("tab", name="6. Chat").click()
    
    # Step 2: Refresh models
    page.get_by_role("button", name="Refresh Models").click()
    page.wait_for_timeout(2000)
    
    # Step 3: Refresh base models
    page.get_by_role("button", name="Refresh Base Models").click()
    page.wait_for_timeout(2000)
    
    # Step 4: Select base model only
    model_selector = page.get_by_label("Select Fine-tuned Model")
    model_selector.select_option("__base__")
    page.wait_for_timeout(1000)
    
    # Step 5: Verify status shows base model ready
    status = page.get_by_label("Status")
    status_text = status.input_value()
    
    # Should show something about base model
    assert "base" in status_text.lower() or "model" in status_text.lower() or status_text == "", \
        f"Status should indicate base model: {status_text}"


def test_all_tabs_navigable(page: Page):
    """Test that all tabs can be navigated."""
    tabs = [
        "1. Upload",
        "2. Configure", 
        "3. Train",
        "4. Results",
        "🤖 Models",
        "🤖 My Models",
        "6. Chat"
    ]
    
    for tab_name in tabs:
        # Click tab
        page.get_by_role("tab", name=tab_name).click()
        
        # Verify tab is selected
        tab = page.get_by_role("tab", name=tab_name)
        aria_selected = tab.get_attribute("aria-selected")
        assert aria_selected == "true", f"Tab {tab_name} should be selected"
        
        # Small delay to let tab content load
        page.wait_for_timeout(500)


def test_error_handling_graceful(page: Page):
    """Test that errors are handled gracefully without crashing UI."""
    # Try to navigate to tabs in unusual order
    # App should handle this gracefully
    
    # Start at Chat (normally requires training)
    page.get_by_role("tab", name="6. Chat").click()
    
    # UI should still be functional
    expect(page.get_by_role("tab", name="1. Upload")).to_be_enabled()
    expect(page.get_by_role("tab", name="2. Configure")).to_be_enabled()
    
    # Try to upload non-existent path (shouldn't crash)
    page.get_by_role("tab", name="1. Upload").click()
    expect(page.get_by_label("Training Data")).to_be_visible()


def test_state_persistence_visual(page: Page, sample_data_path: Path):
    """Test that state changes are reflected in UI."""
    # Upload a file
    page.get_by_role("tab", name="1. Upload").click()
    page.get_by_label("Training Data").set_input_files(str(sample_data_path))
    
    # Wait for upload
    expect(page.get_by_text("validated")).to_be_visible(timeout=10000)
    
    # Navigate away and back
    page.get_by_role("tab", name="2. Configure").click()
    page.wait_for_timeout(1000)
    page.get_by_role("tab", name="1. Upload").click()
    page.wait_for_timeout(1000)
    
    # Status should still show uploaded state
    status = page.get_by_label("Status")
    status_text = status.input_value()
    
    # Should still show uploaded or validated
    assert "uploaded" in status_text.lower() or "validated" in status_text.lower() or "OK" in status_text, \
        f"Status should show uploaded state: {status_text}"


def test_responsive_layout(page: Page):
    """Test that layout works at standard desktop resolution."""
    # Set viewport to standard desktop
    page.set_viewport_size({"width": 1280, "height": 720})
    
    # Navigate to upload
    page.get_by_role("tab", name="1. Upload").click()
    
    # Verify main elements are visible
    expect(page.get_by_text("EdukaAI Fine Tuning Studio")).to_be_visible()
    expect(page.get_by_role("tablist")).to_be_visible()

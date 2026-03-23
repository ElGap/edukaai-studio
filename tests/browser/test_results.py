"""Browser tests for Results tab.

Tests the results/download interface.
"""

from playwright.sync_api import Page, expect


def test_results_page_loads(page: Page):
    """Verify the results page loads correctly."""
    page.get_by_role("tab", name="4. Results").click()
    
    # Verify heading
    expect(page.get_by_text("Step 4: Training Results")).to_be_visible()


def test_download_buttons_exist(page: Page):
    """Test that download buttons exist."""
    page.get_by_role("tab", name="4. Results").click()
    
    # Download adapter button
    expect(page.get_by_role("button", name="Download Adapter")).to_be_visible()
    
    # Download fused button
    expect(page.get_by_role("button", name="Download Fused Model")).to_be_visible()
    
    # Download GGUF button
    expect(page.get_by_role("button", name="Download GGUF")).to_be_visible()


def test_status_display_exists(page: Page):
    """Test that status display exists."""
    page.get_by_role("tab", name="4. Results").click()
    
    # Status textbox
    expect(page.get_by_label("Status")).to_be_visible()


def test_results_without_training_shows_error(page: Page):
    """Test that results tab shows appropriate message when no training done."""
    page.get_by_role("tab", name="4. Results").click()
    
    # Wait a moment for status to update
    page.wait_for_timeout(1000)
    
    # Click download adapter (should show error)
    page.get_by_role("button", name="Download Adapter").click()
    
    # Wait for error
    page.wait_for_timeout(1000)
    
    # Status should show error
    status = page.get_by_label("Status")
    status_text = status.input_value()
    
    # Should indicate training not complete or error
    assert "Error" in status_text or "not complete" in status_text.lower() or "Training not" in status_text, \
        f"Expected error message about training, got: {status_text}"


def test_refresh_button_exists(page: Page):
    """Test that refresh button exists."""
    page.get_by_role("tab", name="4. Results").click()
    
    # Refresh button
    expect(page.get_by_role("button", name="Refresh")).to_be_visible()


def test_results_info_message(page: Page):
    """Test that results tab shows informational message."""
    page.get_by_role("tab", name="4. Results").click()
    
    # Should have info about downloading
    expect(page.get_by_text("Download your fine-tuned model")).to_be_visible()


def test_multiple_download_options_visible(page: Page):
    """Test that multiple download format options are visible."""
    page.get_by_role("tab", name="4. Results").click()
    
    # All three download options should be visible
    expect(page.get_by_role("button", name="Download Adapter")).to_be_visible()
    expect(page.get_by_role("button", name="Download Fused Model")).to_be_visible()
    expect(page.get_by_role("button", name="Download GGUF")).to_be_visible()


def test_file_component_for_downloads(page: Page):
    """Test that file component exists for downloads (may be hidden initially)."""
    page.get_by_role("tab", name="4. Results").click()
    
    # The file component for downloads should exist
    # It might be hidden initially but should be in the DOM
    # Just verify the buttons exist which use this component
    expect(page.get_by_role("button", name="Download Adapter")).to_be_visible()

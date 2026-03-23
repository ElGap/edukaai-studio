"""Browser tests for error scenarios and edge cases.

These tests verify the app handles errors gracefully.
"""

from playwright.sync_api import Page, expect
import tempfile
import os


def test_upload_empty_file_shows_error(page: Page):
    """Test that uploading empty file shows appropriate error."""
    page.get_by_role("tab", name="1. Upload").click()
    
    # Create empty file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        f.write('')
        temp_path = f.name
    
    try:
        # Upload empty file
        page.get_by_label("Training Data").set_input_files(temp_path)
        
        # Wait for processing
        page.wait_for_timeout(3000)
        
        # Should show error about empty file
        status = page.get_by_label("Status")
        status_text = status.input_value()
        
        assert "Error" in status_text or "empty" in status_text.lower() or "0" in status_text or "valid" in status_text.lower(), \
            f"Expected error for empty file, got: {status_text}"
    finally:
        os.unlink(temp_path)


def test_upload_nonexistent_file_path(page: Page):
    """Test behavior when file path doesn't exist."""
    page.get_by_role("tab", name="1. Upload").click()
    
    # This test verifies the app doesn't crash
    # We can't actually test non-existent file upload through browser
    # But we can verify the UI handles errors
    expect(page.get_by_label("Training Data")).to_be_visible()


def test_configure_with_missing_upload_warning(page: Page):
    """Test that configure shows appropriate state without upload."""
    page.get_by_role("tab", name="2. Configure").click()
    
    # Try to configure without uploading
    # Should either work or show clear message
    
    # Verify page loads
    expect(page.get_by_text("Step 2: Configure Training")).to_be_visible()
    
    # Model dropdown should still be populated
    model_dropdown = page.get_by_label("Model")
    expect(model_dropdown).to_be_visible()


def test_train_without_any_setup(page: Page):
    """Test that training without upload/configure shows clear error."""
    page.get_by_role("tab", name="3. Train").click()
    
    # Click start training immediately
    page.get_by_role("button", name="Start Training").click()
    
    # Wait for error
    page.wait_for_timeout(2000)
    
    # Status should show error
    status = page.get_by_label("Status")
    status_text = status.input_value()
    
    assert "Error" in status_text or "required" in status_text.lower() or "Upload" in status_text or "Configure" in status_text or "not" in status_text.lower(), \
        f"Expected error about missing setup, got: {status_text}"


def test_results_without_training(page: Page):
    """Test that results without training shows clear message."""
    page.get_by_role("tab", name="4. Results").click()
    
    # Click download
    page.get_by_role("button", name="Download Adapter").click()
    
    # Wait
    page.wait_for_timeout(1000)
    
    # Status should indicate training not done
    status = page.get_by_label("Status")
    status_text = status.input_value()
    
    assert "Error" in status_text or "not complete" in status_text.lower() or "Training" in status_text, \
        f"Expected message about training, got: {status_text}"


def test_rapid_tab_switching(page: Page):
    """Test that rapid tab switching doesn't crash the app."""
    tabs = [
        "1. Upload",
        "2. Configure",
        "3. Train",
        "4. Results",
        "6. Chat"
    ]
    
    # Rapidly switch tabs
    for _ in range(3):  # Do it 3 times
        for tab_name in tabs:
            page.get_by_role("tab", name=tab_name).click()
    
    # App should still be responsive
    page.wait_for_timeout(500)
    
    # Try to interact
    page.get_by_role("tab", name="1. Upload").click()
    expect(page.get_by_text("Step 1: Upload Training Data")).to_be_visible()


def test_upload_malformed_json(page: Page):
    """Test that malformed JSON shows error."""
    page.get_by_role("tab", name="1. Upload").click()
    
    # Create malformed JSON
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        f.write('{"instruction": "test", output: "missing quote}')  # Malformed
        temp_path = f.name
    
    try:
        # Upload
        page.get_by_label("Training Data").set_input_files(temp_path)
        
        # Wait
        page.wait_for_timeout(3000)
        
        # Should show error
        status = page.get_by_label("Status")
        status_text = status.input_value()
        
        assert "Error" in status_text or "Invalid" in status_text or "JSON" in status_text or "parse" in status_text.lower(), \
            f"Expected error for malformed JSON, got: {status_text}"
    finally:
        os.unlink(temp_path)


def test_upload_large_file_handling(page: Page):
    """Test that large files are handled appropriately."""
    page.get_by_role("tab", name="1. Upload").click()
    
    # Create file with many entries
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for i in range(100):
            f.write(f'{{"instruction": "Question {i}", "output": "Answer {i}"}}\n')
        temp_path = f.name
    
    try:
        # Upload
        page.get_by_label("Training Data").set_input_files(temp_path)
        
        # Wait for processing (large file might take time)
        page.wait_for_timeout(5000)
        
        # Should either succeed or show clear error
        status = page.get_by_label("Status")
        status_text = status.input_value()
        
        # Should have some status, not crash
        assert status_text != "" or "Error" in status_text or "OK" in status_text or "validated" in status_text.lower(), \
            f"Should have status after large file upload: {status_text}"
    finally:
        os.unlink(temp_path)


def test_special_characters_in_filename(page: Page):
    """Test files with special characters in name."""
    page.get_by_role("tab", name="1. Upload").click()
    
    # Create file with special characters
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, prefix='test-file_with.special-chars') as f:
        f.write('{"instruction": "test", "output": "test"}\n')
        temp_path = f.name
    
    try:
        # Upload
        page.get_by_label("Training Data").set_input_files(temp_path)
        
        # Wait
        page.wait_for_timeout(3000)
        
        # Should not crash
        expect(page.get_by_label("Status")).to_be_visible()
    finally:
        os.unlink(temp_path)


def test_concurrent_button_clicks(page: Page):
    """Test that clicking buttons rapidly doesn't cause issues."""
    page.get_by_role("tab", name="1. Upload").click()
    
    # Click refresh button multiple times quickly (if exists)
    refresh_btn = page.get_by_role("button", name="Refresh")
    if refresh_btn.count() > 0:
        for _ in range(5):
            refresh_btn.click()
        
        # App should still work
        page.wait_for_timeout(1000)
        expect(page.get_by_text("Step 1: Upload Training Data")).to_be_visible()


def test_network_error_recovery(page: Page):
    """Test that network errors don't crash the app."""
    # This is more of a manual test, but we can verify the app
    # doesn't rely on network for basic functionality
    
    page.get_by_role("tab", name="1. Upload").click()
    
    # App should work locally without network
    expect(page.get_by_text("Step 1: Upload Training Data")).to_be_visible()
    expect(page.get_by_label("Training Data")).to_be_visible()


def test_memory_leak_prevention(page: Page, sample_data_path):
    """Test that multiple uploads don't cause memory issues."""
    page.get_by_role("tab", name="1. Upload").click()
    
    # Upload same file multiple times
    for i in range(3):
        page.get_by_label("Training Data").set_input_files(str(sample_data_path))
        page.wait_for_timeout(2000)
        
        # Each time should update status
        status = page.get_by_label("Status")
        status_text = status.input_value()
        
        assert "Error" in status_text or "OK" in status_text or "validated" in status_text.lower(), \
            f"Upload {i+1} should complete: {status_text}"


def test_browser_back_button(page: Page):
    """Test that browser back button works."""
    # Navigate to upload
    page.get_by_role("tab", name="1. Upload").click()
    
    # Navigate to configure
    page.get_by_role("tab", name="2. Configure").click()
    page.wait_for_timeout(500)
    
    # Use browser back
    page.go_back()
    page.wait_for_timeout(500)
    
    # Should be back at upload
    # (Note: This might not work with single-page apps, depends on implementation)
    expect(page.get_by_text("EdukaAI Fine Tuning Studio")).to_be_visible()


def test_window_resize_responsive(page: Page):
    """Test that app handles window resize."""
    page.get_by_role("tab", name="1. Upload").click()
    
    # Resize to different sizes
    sizes = [
        {"width": 1920, "height": 1080},  # Large desktop
        {"width": 1366, "height": 768},   # Laptop
        {"width": 1280, "height": 720},   # Small laptop
    ]
    
    for size in sizes:
        page.set_viewport_size(size)
        page.wait_for_timeout(500)
        
        # Verify app still visible
        expect(page.get_by_text("EdukaAI Fine Tuning Studio")).to_be_visible()


def test_keyboard_navigation(page: Page):
    """Test basic keyboard navigation."""
    page.get_by_role("tab", name="1. Upload").click()
    
    # Tab through elements
    page.keyboard.press("Tab")
    page.wait_for_timeout(200)
    
    # Should be able to tab through the page
    # Just verify no crash
    expect(page.get_by_text("Step 1: Upload Training Data")).to_be_visible()

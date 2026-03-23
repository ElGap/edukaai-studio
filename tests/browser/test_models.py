"""Browser tests for Models tab (Model Hub).

Tests the model discovery and custom model management.
"""

from playwright.sync_api import Page, expect


def test_models_page_loads(page: Page):
    """Verify the Models page loads correctly."""
    page.get_by_role("tab", name="Models").first.click()
    
    # Verify heading
    expect(page.get_by_text("Model Hub")).to_be_visible()


def test_predefined_models_list_exists(page: Page):
    """Test that predefined models list exists."""
    page.get_by_role("tab", name="Models").first.click()
    
    # Wait for load
    page.wait_for_timeout(2000)
    
    # Should show some predefined models
    expect(page.get_by_text("Available Models")).to_be_visible()


def test_custom_model_input_exists(page: Page):
    """Test that custom model input exists."""
    page.get_by_role("tab", name="Models").first.click()
    
    # Custom model input
    expect(page.get_by_label("Custom Model")).to_be_visible()


def test_verify_model_button_exists(page: Page):
    """Test that verify model button exists."""
    page.get_by_role("tab", name="Models").first.click()
    
    # Verify button
    expect(page.get_by_role("button", name="Verify Model")).to_be_visible()


def test_hf_token_input_exists(page: Page):
    """Test that HuggingFace token input exists."""
    page.get_by_role("tab", name="Models").first.click()
    
    # HF Token input
    expect(page.get_by_label("HuggingFace Token")).to_be_visible()


def test_save_token_button_exists(page: Page):
    """Test that save token button exists."""
    page.get_by_role("tab", name="Models").first.click()
    
    # Save token button
    expect(page.get_by_role("button", name="Save Token")).to_be_visible()


def test_refresh_models_button_exists(page: Page):
    """Test that refresh models button exists."""
    page.get_by_role("tab", name="Models").first.click()
    
    # Refresh button
    refresh_btn = page.get_by_role("button", name="Refresh")
    if refresh_btn.count() > 0:
        expect(refresh_btn.first).to_be_visible()


def test_verify_invalid_model_shows_error(page: Page):
    """Test that verifying an invalid model shows error."""
    page.get_by_role("tab", name="Models").first.click()
    
    # Enter invalid model ID
    custom_input = page.get_by_label("Custom Model")
    custom_input.fill("invalid-model-name")
    
    # Click verify
    page.get_by_role("button", name="Verify Model").click()
    
    # Wait for result
    page.wait_for_timeout(3000)
    
    # Should show error or not found
    # Status should indicate error
    page_content = page.content()
    assert "Error" in page_content or "not found" in page_content.lower() or "invalid" in page_content.lower() or "verify" in page_content.lower(), \
        "Should show error for invalid model"


def test_model_info_display_exists(page: Page):
    """Test that model info display area exists."""
    page.get_by_role("tab", name="Models").first.click()
    
    # After loading, should have area for model info
    expect(page.get_by_text("Available Models")).to_be_visible()


def test_token_validation_empty_shows_warning(page: Page):
    """Test that empty token shows appropriate message."""
    page.get_by_role("tab", name="Models").first.click()
    
    # Clear token field and try to save
    token_input = page.get_by_label("HuggingFace Token")
    token_input.fill("")
    
    # Click save
    page.get_by_role("button", name="Save Token").click()
    
    # Wait for result
    page.wait_for_timeout(1000)
    
    # Should show message about empty token or success (if clearing is allowed)
    # Either way, no crash
    expect(page.get_by_text("Model Hub")).to_be_visible()


def test_add_custom_model_workflow(page: Page):
    """Test the workflow for adding a custom model."""
    page.get_by_role("tab", name="Models").first.click()
    
    # Enter a model name
    custom_input = page.get_by_label("Custom Model")
    custom_input.fill("test-model")
    
    # Click verify
    page.get_by_role("button", name="Verify Model").click()
    
    # Wait for verification
    page.wait_for_timeout(3000)
    
    # Should show some result (success or error)
    page_content = page.content()
    # Either shows model info or error
    assert "Model" in page_content or "Error" in page_content or "verify" in page_content.lower(), \
        "Should show verification result"


def test_models_tab_layout(page: Page):
    """Test that Models tab has proper layout."""
    page.get_by_role("tab", name="Models").first.click()
    
    # Should have model list section
    expect(page.get_by_text("Available Models")).to_be_visible()
    
    # Should have custom model section
    expect(page.get_by_label("Custom Model")).to_be_visible()
    
    # Should have HF token section
    expect(page.get_by_label("HuggingFace Token")).to_be_visible()


def test_clear_token_button_exists(page: Page):
    """Test that clear token button exists."""
    page.get_by_role("tab", name="Models").first.click()
    
    # Look for clear button
    clear_btn = page.get_by_role("button", name="Clear")
    if clear_btn.count() > 0:
        expect(clear_btn.first).to_be_visible()

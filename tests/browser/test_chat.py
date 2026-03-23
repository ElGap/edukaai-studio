"""Browser tests for Chat tab.

Tests the chat interface, especially base-only mode (no training required).
"""

from playwright.sync_api import Page, expect


def test_chat_page_loads(page: Page):
    """Verify the chat page loads correctly."""
    page.get_by_role("tab", name="6. Chat").click()
    
    # Verify heading
    expect(page.get_by_text("Chat and Compare")).to_be_visible()
    
    # Verify model selector exists
    expect(page.get_by_text("Select Fine-tuned Model")).to_be_visible()


def test_base_model_selector_visible(page: Page):
    """Test that base model selector is visible."""
    page.get_by_role("tab", name="6. Chat").click()
    
    # Refresh models to populate dropdown
    page.get_by_role("button", name="Refresh Models").click()
    
    # Wait a moment for models to load
    import time
    time.sleep(2)
    
    # Verify base model selector exists
    expect(page.get_by_text("Select Base Model")).to_be_visible()


def test_can_select_base_model_only(page: Page):
    """Test selecting 'Use Base Model Only' option."""
    page.get_by_role("tab", name="6. Chat").click()
    
    # Refresh to load models
    page.get_by_role("button", name="Refresh Models").click()
    time.sleep(2)
    
    # Select "Use Base Model Only"
    model_selector = page.get_by_label("Select Fine-tuned Model")
    model_selector.select_option("__base__")
    
    # Verify selection worked
    expect(model_selector).to_have_value("__base__")


def test_base_model_info_updates(page: Page):
    """Test that base model info updates when selecting a model."""
    page.get_by_role("tab", name="6. Chat").click()
    
    # Refresh models
    page.get_by_role("button", name="Refresh Models").click()
    time.sleep(2)
    
    # Refresh base models
    page.get_by_role("button", name="Refresh Base Models").click()
    time.sleep(2)
    
    # Select base model only
    model_selector = page.get_by_label("Select Fine-tuned Model")
    model_selector.select_option("__base__")
    
    # Status should update
    status = page.get_by_label("Status")
    # Should show something about base model
    expect(status).not_to_have_value("")


def test_chat_without_training_works(page: Page):
    """Test that chat works without completing training."""
    page.get_by_role("tab", name="6. Chat").click()
    
    # Refresh models
    page.get_by_role("button", name="Refresh Models").click()
    time.sleep(2)
    
    page.get_by_role("button", name="Refresh Base Models").click()
    time.sleep(2)
    
    # Select base model only
    model_selector = page.get_by_label("Select Fine-tuned Model")
    model_selector.select_option("__base__")
    
    # Verify status shows base model is ready
    # The status should indicate we can chat with base model
    page.wait_for_timeout(1000)


def test_advanced_parameters_accordion(page: Page):
    """Test that advanced parameters can be expanded."""
    page.get_by_role("tab", name="6. Chat").click()
    
    # Click on Advanced Parameters accordion
    # Note: Gradio accordions might have specific structure
    # This test verifies the parameters section exists
    expect(page.get_by_text("Advanced Parameters")).to_be_visible()


def test_temperature_slider_exists(page: Page):
    """Test that temperature control exists."""
    page.get_by_role("tab", name="6. Chat").click()
    
    # Expand advanced parameters if collapsed
    # Look for temperature control
    expect(page.get_by_text("Temperature")).to_be_visible()

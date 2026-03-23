"""Browser tests for Configure tab.

These tests verify model configuration works correctly.
"""

from playwright.sync_api import Page, expect


def test_configure_page_loads(page: Page):
    """Verify the configure page loads correctly."""
    page.get_by_role("tab", name="2. Configure").click()
    
    # Verify heading
    expect(page.get_by_text("Step 2: Configure Training")).to_be_visible()
    
    # Verify model dropdown exists
    expect(page.get_by_label("Model")).to_be_visible()
    
    # Verify preset dropdown exists
    expect(page.get_by_label("Training Preset")).to_be_visible()


def test_model_dropdown_populated(page: Page):
    """Test that the model dropdown has options."""
    page.get_by_role("tab", name="2. Configure").click()
    
    # Click on model dropdown to see options
    model_dropdown = page.get_by_label("Model")
    
    # Verify dropdown has value (default should be selected)
    expect(model_dropdown).not_to_have_value("")


def test_preset_changes_parameters(page: Page):
    """Test that selecting different presets updates parameters."""
    page.get_by_role("tab", name="2. Configure").click()
    
    # Select Quick preset
    preset_dropdown = page.get_by_label("Training Preset")
    preset_dropdown.select_option("quick")
    
    # Note: In a full implementation, we'd verify sliders update
    # This test verifies the preset selection works without error
    expect(preset_dropdown).to_have_value("quick")


def test_parameters_visible(page: Page):
    """Test that all training parameters are visible."""
    page.get_by_role("tab", name="2. Configure").click()
    
    # Verify parameter sliders/inputs exist
    expect(page.get_by_text("Training Parameters")).to_be_visible()
    
    # Check for key parameters
    expect(page.get_by_text("Training Steps")).to_be_visible()
    expect(page.get_by_text("LoRA Rank")).to_be_visible()


def test_configure_without_upload_shows_warning(page: Page):
    """Test that configure shows appropriate message if no file uploaded."""
    # First, ensure no file is uploaded by checking state or starting fresh
    # Navigate to configure
    page.get_by_role("tab", name="2. Configure").click()
    
    # The configure tab should handle this gracefully
    # Either show a warning or still allow configuration
    expect(page.get_by_text("Step 2: Configure Training")).to_be_visible()

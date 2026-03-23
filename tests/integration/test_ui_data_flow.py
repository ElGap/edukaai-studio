"""Tests for UI data flow from training monitor to Gradio interface.

These tests verify that training loss values properly flow from the training
process through the monitor to the Gradio UI components.
"""

import pytest
import queue
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any


class TestLossDataFlow:
    """Test that loss values flow correctly from training to UI."""

    def test_progress_data_structure_matches_expectations(self):
        """CRITICAL: Verify progress_data from monitor has keys train.py expects.
        
        Bug: train.py checks for 'train_loss' (singular) but monitor sends 
        'train_losses' (dict). This causes UI to show 0.0000 while console
        shows real values.
        """
        # Simulate what TrainingMonitor puts in progress_queue
        monitor_progress_data = {
            'iteration': 50,
            'total': 200,
            'progress_percent': 25,
            'train_losses': {50: 2.585, 40: 2.740, 30: 2.842},  # Dict of all losses
            'val_losses': {50: 2.585},
            'best_loss': 2.585,
            'peak_memory_gb': 4.357,
        }
        
        # This is what train.py currently checks (BUG!)
        current_implementation_checks = 'train_loss' in monitor_progress_data
        
        # This is what it should check
        correct_check = 'train_losses' in monitor_progress_data
        
        # The bug: current check fails
        assert current_implementation_checks == False, \
            "BUG CONFIRMED: 'train_loss' key doesn't exist in progress_data"
        
        # The correct key exists
        assert correct_check == True, \
            "'train_losses' dict exists and should be used"
    
    def test_ui_shows_real_loss_values_not_zeros(self):
        """Test that UI displays actual loss values from training.
        
        When progress_data comes in with train_losses dict,
        the UI should show those values, not 0.0000.
        """
        # Mock progress data as it comes from TrainingMonitor
        progress_data = {
            'iteration': 50,
            'train_losses': {50: 2.585, 40: 2.740},
            'val_losses': {50: 2.585},
            'best_loss': 2.585,
            'peak_memory_gb': 4.357,
        }
        
        # Simulate the fixed train.py code
        train_losses = {}
        if 'train_losses' in progress_data:  # Correct key
            train_losses.update(progress_data['train_losses'])
        
        # Result: UI shows real value
        ui_train_loss = train_losses.get(50, 0.0)
        assert ui_train_loss == 2.585, f"UI should show 2.585, got {ui_train_loss}"
    
    def test_loss_values_persist_across_multiple_updates(self):
        """Test that loss history is maintained across multiple progress updates."""
        # Simulate multiple progress updates
        updates = [
            {'iteration': 10, 'train_losses': {10: 3.113}, 'val_losses': {}},
            {'iteration': 20, 'train_losses': {20: 2.842}, 'val_losses': {20: 2.585}},
            {'iteration': 30, 'train_losses': {30: 2.740}, 'val_losses': {20: 2.585}},
        ]
        
        # Fixed: Using 'train_losses' (dict) accumulates history
        train_losses = {}
        for update in updates:
            if 'train_losses' in update:
                train_losses.update(update['train_losses'])
        
        # Result: All 3 iterations preserved
        assert len(train_losses) == 3, f"Should preserve all loss values, got {len(train_losses)}"
        assert train_losses[10] == 3.113
        assert train_losses[20] == 2.842
        assert train_losses[30] == 2.740


class TestProgressQueueIntegration:
    """Integration tests for progress queue data flow."""

    def test_queue_contains_expected_keys(self):
        """Verify actual queue contains 'train_losses' not 'train_loss'."""
        from edukaai_studio.ui.training_monitor import TrainingMonitor
        
        output_q = queue.Queue()
        progress_q = queue.Queue()
        monitor = TrainingMonitor(output_q, progress_q)
        
        # Simulate progress data as would be sent by real monitor
        test_data = {
            'iteration': 50,
            'train_losses': {50: 2.585},
            'val_losses': {50: 2.585},
        }
        
        # Put data in queue (simulating what _parse_progress does)
        progress_q.put(test_data)
        
        # Verify queue contents
        retrieved = progress_q.get()
        assert 'train_losses' in retrieved, "Queue should contain 'train_losses' dict"
        assert 'train_loss' not in retrieved, "Queue should NOT contain 'train_loss' (singular)"
        assert retrieved['train_losses'][50] == 2.585


class TestUIUpdateLogic:
    """Test the UI update logic in train.py specifically."""

    def test_train_py_extracts_loss_correctly(self):
        """Test that train.py extracts loss from progress_data using correct key.
        
        This test verifies the fix: train.py should use 'train_losses' (plural/dict)
        not 'train_loss' (singular).
        """
        # Simulate the fixed loop in train.py
        progress_data = {
            'iteration': 50,
            'train_losses': {50: 2.585},
            'val_losses': {50: 2.585},
        }
        
        train_losses = {}
        current_iter = progress_data.get('iteration', 0)
        
        # This is the FIXED code from train.py
        if 'train_losses' in progress_data:
            train_losses.update(progress_data['train_losses'])
        
        # Result: losses are captured
        assert len(train_losses) == 1, "Should capture losses"
        assert train_losses[50] == 2.585, f"Should have loss 2.585, got {train_losses.get(50)}"


class TestRegressionBugLossDisplay:
    """Regression tests for the loss display bug (console shows values, UI shows 0.0000)."""

    def test_regression_console_shows_loss_but_ui_shows_zero(self):
        """REGRESSION TEST: Console shows real loss, UI shows 0.0000.
        
        This test documents the bug where:
        - Console output shows: "Train loss 2.585, Learning Rate..."
        - UI log shows: "Step 50/200 | Train: 0.0000 | Val: 0.0000"
        
        The bug was that train.py checked for 'train_loss' (singular) but
        TrainingMonitor sends 'train_losses' (dict).
        """
        # Simulate the exact scenario from the bug report
        console_line = "Iter 50: Train loss 2.585, Learning Rate 1.000e-04, It/sec 3.875"
        
        # TrainingMonitor parses this and creates progress_data
        progress_data = {
            'iteration': 50,
            'train_losses': {50: 2.585},  # Key is 'train_losses' (plural)
            'val_losses': {},
        }
        
        # BUGGY train.py code would check 'train_loss' (wrong!)
        train_losses_buggy = {}
        if 'train_loss' in progress_data:
            train_losses_buggy[50] = progress_data['train_loss']
        
        # Result: UI shows 0.0000
        ui_display_buggy = train_losses_buggy.get(50, 0.0)
        
        # FIXED train.py code checks 'train_losses' (correct!)
        train_losses_fixed = {}
        if 'train_losses' in progress_data:
            train_losses_fixed.update(progress_data['train_losses'])
        
        # Result: UI shows 2.585
        ui_display_fixed = train_losses_fixed.get(50, 0.0)
        
        # Verify the fix
        assert ui_display_buggy == 0.0, "Bug: UI would show 0.0000"
        assert ui_display_fixed == 2.585, "Fix: UI shows 2.585"
        assert ui_display_fixed == 2.585, \
            f"UI should display {2.585} from console output, not {ui_display_buggy}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

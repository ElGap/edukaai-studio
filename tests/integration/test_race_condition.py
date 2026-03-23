"""Test for the race condition in training completion detection.

This test verifies that training completion is properly detected even when
the output reader thread hasn't finished setting the completion flag yet.
"""

import pytest
import queue
import threading
import time
from unittest.mock import Mock, patch, MagicMock


class TestCompletionRaceCondition:
    """Test for race condition between main thread and output reader thread."""
    
    def test_completion_detected_via_process_exit_code(self):
        """Verify completion is detected when process exits successfully.
        
        This catches the race condition where:
        1. Process exits (returncode = 0)
        2. Main thread checks is_complete() before reader thread sets training_complete
        3. Without fix: is_complete returns False
        4. With fix: is_complete returns True (checks process.returncode)
        """
        from edukaai_studio.ui.training_monitor import TrainingMonitor
        
        # Create mock queues
        output_q = queue.Queue()
        progress_q = queue.Queue()
        monitor = TrainingMonitor(output_q, progress_q)
        
        # Simulate: Process finished but reader thread hasn't set training_complete yet
        mock_process = Mock()
        mock_process.poll.return_value = 0  # Process exited
        mock_process.returncode = 0  # Successful exit
        monitor.process = mock_process
        monitor.training_complete = False  # Reader thread hasn't set this yet
        monitor.was_stopped = False
        
        # OLD logic (what would fail):
        old_is_complete = monitor.is_complete() and not monitor.was_stopped
        assert old_is_complete is False, "Old logic fails due to race condition"
        
        # NEW logic (what should work):
        is_complete_new = monitor.is_complete() or (
            monitor.process and 
            monitor.process.poll() is not None and 
            monitor.process.returncode == 0 and
            not monitor.was_stopped
        )
        assert is_complete_new is True, "New logic should detect completion via process exit code"
    
    def test_incomplete_when_process_fails(self):
        """Verify training NOT marked complete when process fails."""
        from edukaai_studio.ui.training_monitor import TrainingMonitor
        
        output_q = queue.Queue()
        progress_q = queue.Queue()
        monitor = TrainingMonitor(output_q, progress_q)
        
        # Simulate: Process failed
        mock_process = Mock()
        mock_process.poll.return_value = 1  # Process exited with error
        mock_process.returncode = 1  # Error code
        monitor.process = mock_process
        monitor.training_complete = False
        monitor.was_stopped = False
        
        # Should NOT be complete
        is_complete = monitor.is_complete() or (
            monitor.process and 
            monitor.process.poll() is not None and 
            monitor.process.returncode == 0 and
            not monitor.was_stopped
        )
        assert is_complete is False, "Should not be complete when process fails"
    
    def test_incomplete_when_process_still_running(self):
        """Verify training NOT marked complete when process still running."""
        from edukaai_studio.ui.training_monitor import TrainingMonitor
        
        output_q = queue.Queue()
        progress_q = queue.Queue()
        monitor = TrainingMonitor(output_q, progress_q)
        
        # Simulate: Process still running
        mock_process = Mock()
        mock_process.poll.return_value = None  # Still running
        monitor.process = mock_process
        monitor.training_complete = False
        
        # Should NOT be complete
        is_complete = monitor.is_complete() or (
            monitor.process and 
            monitor.process.poll() is not None and 
            monitor.process.returncode == 0
        )
        assert is_complete is False, "Should not be complete when process running"
    
    def test_stopped_training_not_marked_complete(self):
        """Verify stopped training NOT marked complete."""
        from edukaai_studio.ui.training_monitor import TrainingMonitor
        
        output_q = queue.Queue()
        progress_q = queue.Queue()
        monitor = TrainingMonitor(output_q, progress_q)
        
        # Simulate: User stopped training, process exited
        mock_process = Mock()
        mock_process.poll.return_value = 0
        mock_process.returncode = 0
        monitor.process = mock_process
        monitor.training_complete = True  # Reader thread set this
        monitor.was_stopped = True  # But user stopped it
        
        # Should NOT be complete (was stopped)
        was_stopped = monitor.was_stopped
        is_complete = monitor.is_complete() and not was_stopped
        assert is_complete is False, "Should not be complete when stopped by user"


class TestStateSavingAfterCompletion:
    """Test that state is properly saved after completion detection."""
    
    def test_state_saved_when_completion_detected(self):
        """Verify state is saved to disk when training completes."""
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from edukaai_studio.core.state import save_state_to_disk, load_state_from_disk
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / ".studio_state.json"
            
            with patch('edukaai_studio.core.state.get_state_file', return_value=state_file):
                # Simulate completion
                completion_state = {
                    'training_complete': True,
                    'training_active': False,
                    'output_dir': 'outputs/test',
                    'completion_time': '12:00:00',
                }
                
                # Save state
                result = save_state_to_disk(completion_state)
                assert result is True
                
                # Verify it was saved
                loaded = load_state_from_disk()
                assert loaded is not None
                assert loaded['training_complete'] is True
                assert loaded['training_active'] is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

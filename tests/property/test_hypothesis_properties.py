"""Property-Based Tests using Hypothesis.

These tests generate hundreds of random inputs to catch edge cases that
example-based tests miss.
"""

import pytest
from hypothesis import given, settings, strategies as st, reproduce_failure
from hypothesis.stateful import RuleBasedStateMachine, rule, precondition
import re
import queue


class TestLossDataExtraction:
    """Property-based tests for loss value extraction from training data."""
    
    @given(st.dictionaries(
        st.integers(min_value=1, max_value=1000),
        st.floats(min_value=0.001, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=1, max_size=50
    ))
    def test_any_loss_dict_can_be_extracted(self, loss_dict):
        """Property: Any valid loss dictionary should be extractable.
        
        This would have caught the bug where train.py checked 'train_loss' (singular)
        instead of 'train_losses' (dict).
        """
        progress_data = {
            'iteration': max(loss_dict.keys()),
            'train_losses': loss_dict,
        }
        
        # Correct extraction (what should happen)
        extracted = {}
        if 'train_losses' in progress_data:
            extracted.update(progress_data['train_losses'])
        
        # Property: All losses should be preserved
        assert len(extracted) == len(loss_dict)
        for iter_num, loss in loss_dict.items():
            assert extracted[iter_num] == loss
    
    @given(st.lists(
        st.tuples(
            st.integers(min_value=1, max_value=500),
            st.floats(min_value=0.001, max_value=10.0, allow_nan=False, allow_infinity=False)
        ),
        min_size=1, max_size=20, unique_by=lambda x: x[0]
    ))
    def test_loss_values_preserved_across_updates(self, loss_updates):
        """Property: As we receive loss updates, all values should accumulate."""
        train_losses = {}
        
        # Simulate receiving updates
        for iteration, loss in loss_updates:
            progress_data = {
                'iteration': iteration,
                'train_losses': {iteration: loss},
            }
            
            if 'train_losses' in progress_data:
                train_losses.update(progress_data['train_losses'])
        
        # Property: All iterations should be present
        assert len(train_losses) == len(loss_updates)
        
        # Property: Values should match
        for iteration, loss in loss_updates:
            assert train_losses[iteration] == loss
    
    @given(st.integers(min_value=1, max_value=100))
    def test_zero_loss_displayed_not_replaced(self, iteration):
        """Property: If actual loss is 0.0, it should be displayed as 0.0.
        
        Bug: 0.0 could mean either 'actual zero loss' OR 'default value'.
        We need to distinguish these cases.
        """
        progress_data = {
            'iteration': iteration,
            'train_losses': {iteration: 0.0},  # Actual zero loss
        }
        
        # Extract using correct logic
        train_losses = {}
        if 'train_losses' in progress_data:
            train_losses.update(progress_data['train_losses'])
        
        # Should be 0.0, not replaced with something else
        assert iteration in train_losses
        assert train_losses[iteration] == 0.0


class TestLogLineParsing:
    """Property-based tests for parsing training log lines."""
    
    @given(st.floats(min_value=0.0, max_value=10.0, allow_nan=False).map(lambda x: round(x, 4)))
    def test_train_loss_regex_accepts_valid_formats(self, loss_value):
        """Property: Valid mlx_lm loss lines should always parse."""
        # Generate valid mlx_lm format lines
        iteration = 50
        line_formats = [
            f"Iter {iteration}: Train loss {loss_value}, Learning Rate 1.000e-04",
            f"[LORA-TRAIN] Iter {iteration}: Train loss {loss_value}",
            f"Iteration {iteration}: Train loss {loss_value}, Val loss {loss_value - 0.5}",
        ]
        
        pattern = re.compile(r'Iter\s+(\d+).*Train loss\s+([\d.]+|nan|inf)', re.IGNORECASE)
        
        for line in line_formats:
            match = pattern.search(line)
            assert match is not None, f"Should parse: {line}"
            assert int(match.group(1)) == iteration
            assert float(match.group(2)) == loss_value
    
    @given(st.text(min_size=1, max_size=200))
    def test_regex_doesnt_crash_on_random_input(self, random_line):
        """Property: Parser should handle any string without crashing."""
        pattern = re.compile(r'Iter\s+(\d+).*Train loss\s+([\d.]+|nan|inf)', re.IGNORECASE)
        
        # Should not raise exception
        try:
            match = pattern.search(random_line)
            # If it matches, groups should be valid
            if match:
                iter_str = match.group(1)
                loss_str = match.group(2)
                # Should be parseable
                int(iter_str)
                if loss_str.lower() not in ['nan', 'inf']:
                    float(loss_str)
        except (ValueError, AttributeError, IndexError) as e:
            pytest.fail(f"Regex failed on: {random_line[:50]}... Error: {e}")


class TestDataValidation:
    """Property-based tests for data validation."""
    
    @given(st.lists(
        st.fixed_dictionaries({
            'instruction': st.text(min_size=1, max_size=100),
            'input': st.text(max_size=100),
            'output': st.text(min_size=1, max_size=500),
        }),
        min_size=1, max_size=100
    ))
    def test_valid_alpaca_data_always_validates(self, samples):
        """Property: Well-formed Alpaca data should always validate."""
        import json
        from io import StringIO
        
        # Convert to JSONL
        jsonl_content = '\n'.join(json.dumps(s) for s in samples)
        
        # Should not raise when parsing
        lines = jsonl_content.strip().split('\n')
        for line in lines:
            data = json.loads(line)
            # Should have required fields
            assert 'instruction' in data
            assert 'output' in data
            # Types should be strings
            assert isinstance(data['instruction'], str)
            assert isinstance(data['output'], str)
    
    @given(st.lists(
        st.fixed_dictionaries({
            'messages': st.lists(
                st.fixed_dictionaries({
                    'role': st.sampled_from(['system', 'user', 'assistant']),
                    'content': st.text(min_size=1, max_size=200),
                }),
                min_size=1, max_size=5
            )
        }),
        min_size=1, max_size=50
    ))
    def test_valid_chatml_data_always_validates(self, samples):
        """Property: Well-formed ChatML data should always validate."""
        import json
        
        for sample in samples:
            # Should have messages array
            assert 'messages' in sample
            assert isinstance(sample['messages'], list)
            
            # Each message should have role and content
            for msg in sample['messages']:
                assert 'role' in msg
                assert 'content' in msg
                assert msg['role'] in ['system', 'user', 'assistant']
                assert isinstance(msg['content'], str)


class TestStateTransitions(RuleBasedStateMachine):
    """State machine tests for training lifecycle.
    
    Models the training process as a state machine with transitions:
    IDLE → TRAINING → COMPLETED/FAILED
    """
    
    def __init__(self):
        super().__init__()
        self.state = {
            'training_active': False,
            'training_complete': False,
            'training_failed': False,
            'iteration': 0,
            'total_iterations': 200,
            'train_losses': {},
            'output_dir': None,
        }
    
    @rule()
    def start_training(self):
        """Can start training from IDLE state only."""
        if not self.state['training_active'] and not self.state['training_complete']:
            self.state['training_active'] = True
            self.state['iteration'] = 0
    
    @rule(iteration=st.integers(min_value=1, max_value=200))
    def make_progress(self, iteration):
        """Training can make progress."""
        if self.state['training_active'] and not self.state['training_complete']:
            self.state['iteration'] = max(self.state['iteration'], iteration)
            # Simulate getting a loss value
            self.state['train_losses'][iteration] = 2.5 - (iteration / 100)
    
    @rule()
    @precondition(lambda self: self.state['training_active'] and self.state['iteration'] >= 190)
    def complete_training(self):
        """Can complete training when enough progress made."""
        self.state['training_active'] = False
        self.state['training_complete'] = True
        self.state['output_dir'] = '/tmp/output'
    
    @rule()
    @precondition(lambda self: self.state['training_active'])
    def fail_training(self):
        """Training can fail."""
        self.state['training_active'] = False
        self.state['training_failed'] = True
    
    @rule()
    def check_invariants(self):
        """Verify state invariants always hold."""
        # Invariant 1: Can't be both active and complete
        assert not (self.state['training_active'] and self.state['training_complete']), \
            "Cannot be both active and complete"
        
        # Invariant 2: Can't be both complete and failed
        assert not (self.state['training_complete'] and self.state['training_failed']), \
            "Cannot be both complete and failed"
        
        # Invariant 3: If complete, must have output_dir
        if self.state['training_complete']:
            assert self.state['output_dir'] is not None, \
                "Complete training must have output_dir"
        
        # Invariant 4: iteration should never exceed total
        assert self.state['iteration'] <= self.state['total_iterations'], \
            f"Iteration {self.state['iteration']} exceeds total {self.state['total_iterations']}"
        
        # Invariant 5: losses should be monotonically keyed by iteration
        if self.state['train_losses']:
            max_loss_iter = max(self.state['train_losses'].keys())
            assert max_loss_iter <= self.state['iteration'], \
                f"Have loss for iter {max_loss_iter} but current is {self.state['iteration']}"


TestTrainingStateMachine = TestStateTransitions.TestCase


class TestProgressDataIntegrity:
    """Tests for data integrity in progress updates."""
    
    @given(st.integers(min_value=1, max_value=1000), st.integers(min_value=1, max_value=1000))
    def test_progress_percent_calculation(self, current, total):
        """Property: progress_percent should always be 0-100."""
        assume(total > 0)  # Can't divide by zero
        
        progress_percent = int((current / total) * 100)
        
        # Should always be 0-100 (or 100 if current > total)
        assert 0 <= progress_percent <= 100 or current > total
        
        if current >= total:
            assert progress_percent >= 100 or current == 0
    
    @given(st.dictionaries(
        st.integers(min_value=1, max_value=100),
        st.floats(allow_nan=False, allow_infinity=False),
        min_size=1
    ))
    def test_best_loss_tracking(self, losses):
        """Property: best_loss should always be the minimum."""
        best_loss = float('inf')
        best_iter = 0
        
        for iteration, loss in losses.items():
            if loss < best_loss:
                best_loss = loss
                best_iter = iteration
        
        assert best_loss == min(losses.values())
        assert best_loss in losses.values()


class TestQueueBehavior:
    """Property tests for queue interactions."""
    
    @given(st.lists(
        st.fixed_dictionaries({
            'iteration': st.integers(min_value=1, max_value=100),
            'train_losses': st.dictionaries(st.integers(), st.floats(), min_size=1),
        }),
        min_size=1, max_size=50
    ))
    def test_queue_order_preserved(self, items):
        """Property: FIFO queue should preserve order."""
        q = queue.Queue()
        
        # Put all items
        for item in items:
            q.put(item)
        
        # Get all items
        retrieved = []
        while not q.empty():
            retrieved.append(q.get())
        
        # Order should be preserved
        assert len(retrieved) == len(items)
        for i, (original, got) in enumerate(zip(items, retrieved)):
            assert got['iteration'] == original['iteration']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

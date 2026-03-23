"""Test the test suite on a file with known bugs."""
import sys
from pathlib import Path

# Import the test functions
sys.path.insert(0, '/Users/developer/Projects/edukaai-studio')
from tests.test_realistic import (
    find_deprecated_gradio_calls,
    find_mismatched_yield_outputs,
    find_broken_yield_patterns
)

# Test file with intentional bugs
test_file = Path('/tmp/test_deprecated.py')

print("=" * 60)
print("Testing bug detection on intentional bad code")
print("=" * 60)

print("\n1. Testing deprecated Gradio calls detection:")
issues = find_deprecated_gradio_calls(test_file)
if issues:
    print(f"✅ Found {len(issues)} deprecated calls:")
    for line, content, suggestion in issues:
        print(f"   Line {line}: {content[:50]}")
        print(f"   -> {suggestion}")
else:
    print("❌ FAILED: Should have found deprecated calls")

print("\n2. Testing mismatched yield detection:")
yield_issues = find_mismatched_yield_outputs(test_file)
if yield_issues:
    print(f"✅ Found {len(yield_issues)} mismatched yields:")
    for line, func_name, details in yield_issues:
        print(f"   Line {line}: {func_name}")
        print(f"   -> {details}")
else:
    print("❌ FAILED: Should have found mismatched yields")

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)

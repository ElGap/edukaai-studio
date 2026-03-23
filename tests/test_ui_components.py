"""Test suite for EdukaAI Studio UI components.

This module provides tests to catch common Gradio API issues and other
potential problems before they cause runtime errors.
"""

import ast
import sys
from pathlib import Path
from typing import List, Tuple


def find_deprecated_gradio_calls(file_path: Path) -> List[Tuple[int, str, str]]:
    """Find deprecated Gradio API calls in a Python file.
    
    Checks for:
    - gr.Dropdown.update() -> should be gr.update()
    - gr.Textbox.update() -> should be gr.update()
    - gr.Button.update() -> should be gr.update()
    - gr.Slider.update() -> should be gr.update()
    - gr.File.update() -> should be gr.update()
    
    Args:
        file_path: Path to Python file to check
        
    Returns:
        List of (line_number, line_content, suggestion) tuples
    """
    issues = []
    deprecated_patterns = [
        ('Dropdown', 'gr.Dropdown.update'),
        ('Textbox', 'gr.Textbox.update'),
        ('Button', 'gr.Button.update'),
        ('Slider', 'gr.Slider.update'),
        ('File', 'gr.File.update'),
        ('Plot', 'gr.Plot.update'),
        ('HTML', 'gr.HTML.update'),
        ('Markdown', 'gr.Markdown.update'),
        ('Code', 'gr.Code.update'),
        ('Image', 'gr.Image.update'),
        ('Audio', 'gr.Audio.update'),
        ('Video', 'gr.Video.update'),
        ('Dataframe', 'gr.Dataframe.update'),
        ('JSON', 'gr.JSON.update'),
    ]
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            for component, pattern in deprecated_patterns:
                if pattern in line:
                    issues.append((
                        i,
                        line.strip(),
                        f"Use gr.update() instead of {pattern}()"
                    ))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return issues


def find_mismatched_yield_outputs(file_path: Path) -> List[Tuple[int, str, str]]:
    """Find generator functions that might have mismatched yield outputs.
    
    This checks if yields in a generator function have consistent number of outputs.
    
    Args:
        file_path: Path to Python file to check
        
    Returns:
        List of (line_number, line_content, suggestion) tuples
    """
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if it's a generator function
                has_yield = False
                yield_counts = []
                
                for subnode in ast.walk(node):
                    if isinstance(subnode, ast.Yield):
                        has_yield = True
                        # Count elements in the yielded list/tuple
                        if isinstance(subnode.value, (ast.List, ast.Tuple)):
                            count = len(subnode.value.elts)
                            yield_counts.append((subnode.lineno, count))
                
                if has_yield and yield_counts:
                    # Check if all yields have same count
                    counts = [c for _, c in yield_counts]
                    if len(set(counts)) > 1:
                        issues.append((
                            node.lineno,
                            f"Function '{node.name}' has inconsistent yield counts: {counts}",
                            "All yields should return the same number of values"
                        ))
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    
    return issues


def find_missing_state_updates(file_path: Path) -> List[Tuple[int, str, str]]:
    """Find places where state might not be properly updated.
    
    Args:
        file_path: Path to Python file to check
        
    Returns:
        List of (line_number, line_content, suggestion) tuples
    """
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check for common patterns where state should be updated but might not be
            if 'current_state.get(' in line and 'current_state = {' not in lines[max(0, i-3):i+1]:
                # Check if we're in a function that should update state
                issues.append((
                    i,
                    line.strip(),
                    "Check if state needs to be updated after .get() call"
                ))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return issues


def run_all_tests(src_dir: Path = None) -> bool:
    """Run all tests on the EdukaAI Studio codebase.
    
    Args:
        src_dir: Source directory to test (defaults to edukaai_studio)
        
    Returns:
        True if all tests pass, False otherwise
    """
    if src_dir is None:
        # Get the src directory relative to tests directory
        src_dir = Path(__file__).parent.parent / "src" / "edukaai_studio"
    
    all_passed = True
    total_issues = 0
    
    print("=" * 70)
    print("EdukaAI Studio UI Test Suite")
    print("=" * 70)
    
    # Find all Python files in ui/tabs
    tabs_dir = src_dir / "ui" / "tabs"
    if not tabs_dir.exists():
        print(f"❌ Tabs directory not found: {tabs_dir}")
        return False
    
    python_files = list(tabs_dir.glob("*.py"))
    
    print(f"\nFound {len(python_files)} Python files to test\n")
    
    for file_path in sorted(python_files):
        print(f"Testing: {file_path.name}")
        file_issues = []
        
        # Test 1: Deprecated Gradio calls
        gradio_issues = find_deprecated_gradio_calls(file_path)
        for line, content, suggestion in gradio_issues:
            file_issues.append(("DEPRECATED", line, content, suggestion))
        
        # Test 2: Mismatched yield outputs
        yield_issues = find_mismatched_yield_outputs(file_path)
        for line, content, suggestion in yield_issues:
            file_issues.append(("YIELD_MISMATCH", line, content, suggestion))
        
        # Test 3: Missing state updates (informational only)
        # These are often false positives, so we just log them as warnings
        state_issues = find_missing_state_updates(file_path)
        if state_issues:
            print(f"  ⚠️  {len(state_issues)} potential state check(s) (info only):")
            for line, content, suggestion in state_issues[:3]:  # Show first 3 only
                if len(content) > 60:
                    content = content[:57] + "..."
                print(f"    Line {line}: {content}")
        
        if file_issues:
            all_passed = False
            total_issues += len(file_issues)
            print(f"  ❌ Found {len(file_issues)} issue(s):")
            for issue_type, line, content, suggestion in file_issues:
                print(f"    Line {line}: [{issue_type}] {suggestion}")
                if len(content) > 80:
                    content = content[:77] + "..."
                print(f"      {content}")
        else:
            print(f"  ✅ All tests passed")
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ Critical tests passed! No deprecated API calls found.")
        print("ℹ️  Note: Some informational warnings may be shown above.")
    else:
        print(f"❌ Found {total_issues} critical issue(s) with deprecated API calls.")
        print("   Please fix these issues to ensure Gradio 6.x compatibility.")
    print("=" * 70)
    
    return all_passed


def test_specific_file(file_path: str) -> bool:
    """Test a specific file.
    
    Args:
        file_path: Path to file to test
        
    Returns:
        True if tests pass, False otherwise
    """
    path = Path(file_path)
    if not path.exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    print(f"\nTesting specific file: {path.name}\n")
    
    issues = []
    
    # Run all tests on this file
    gradio_issues = find_deprecated_gradio_calls(path)
    for line, content, suggestion in gradio_issues:
        issues.append(("DEPRECATED", line, content, suggestion))
    
    yield_issues = find_mismatched_yield_outputs(path)
    for line, content, suggestion in yield_issues:
        issues.append(("YIELD_MISMATCH", line, content, suggestion))
    
    if issues:
        print(f"❌ Found {len(issues)} issue(s):")
        for issue_type, line, content, suggestion in issues:
            print(f"\n  Line {line}: [{issue_type}]")
            print(f"  Code: {content[:80]}")
            print(f"  Fix:  {suggestion}")
        return False
    else:
        print(f"✅ All tests passed for {path.name}")
        return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test EdukaAI Studio UI components")
    parser.add_argument(
        "--file",
        type=str,
        help="Test a specific file (relative to src/edukaai_studio/ui/tabs/)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test all files in ui/tabs directory"
    )
    
    args = parser.parse_args()
    
    if args.file:
        # Test specific file
        file_path = Path(__file__).parent.parent / "src" / "edukaai_studio" / "ui" / "tabs" / args.file
        success = test_specific_file(str(file_path))
        sys.exit(0 if success else 1)
    elif args.all or True:  # Default to testing all
        success = run_all_tests()
        sys.exit(0 if success else 1)

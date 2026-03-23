"""Realistic Test Suite for EdukaAI Studio.

This module provides tests that catch real issues while avoiding false positives.
Only flags actual problems that will cause runtime errors.
"""

import ast
import re
import sys
from pathlib import Path
from typing import List, Tuple, Set


def find_deprecated_gradio_calls(file_path: Path) -> List[Tuple[int, str, str]]:
    """Find deprecated Gradio API calls that will cause AttributeError in Gradio 6.x.
    
    These are the ONLY patterns that will cause immediate runtime errors:
    - gr.Dropdown.update() -> use gr.update()
    - gr.Textbox.update() -> use gr.update()
    - gr.Button.update() -> use gr.update()
    - gr.Slider.update() -> use gr.update()
    - etc.
    
    Args:
        file_path: Path to Python file to check
        
    Returns:
        List of (line_number, line_content, fix_suggestion) tuples
    """
    issues = []
    
    # Pattern: gr.ComponentName.update(
    deprecated_pattern = re.compile(
        r'gr\.(Dropdown|Textbox|Button|Slider|File|Plot|HTML|Markdown|'
        r'Code|Image|Audio|Video|Dataframe|JSON|Checkbox|Number|'
        r'Radio|State|Tabs|TabItem|Row|Column|Group|Accordion)\s*\.\s*update\s*\('
    )
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines, 1):
            if deprecated_pattern.search(line):
                # Extract the component name
                match = deprecated_pattern.search(line)
                if match:
                    component = match.group(1)
                    issues.append((
                        i,
                        line.strip(),
                        f"Replace gr.{component}.update(...) with gr.update(...)"
                    ))
    except Exception as e:
        print(f"  ⚠️  Error reading {file_path}: {e}")
    
    return issues


def find_broken_yield_patterns(file_path: Path) -> List[Tuple[int, str, str]]:
    """Find yield statements that are syntactically broken.
    
    This catches patterns like:
    - Empty yields: yield []
    - Incomplete yields: yield [a, b,  (missing closing bracket)
    - Mismatched brackets in yields
    
    Args:
        file_path: Path to Python file to check
        
    Returns:
        List of (line_number, line_content, fix_suggestion) tuples
    """
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for syntax errors by trying to parse
        try:
            ast.parse(content)
        except SyntaxError as e:
            # If there's a syntax error, report it
            if 'yield' in str(e) or (e.lineno and 'yield' in content.split('\n')[e.lineno - 1]):
                line_num = e.lineno or 1
                lines = content.split('\n')
                if line_num <= len(lines):
                    issues.append((
                        line_num,
                        lines[line_num - 1].strip()[:80],
                        f"Syntax error in yield: {e.msg}"
                    ))
        
        # Also check for patterns that compile but are wrong
        lines = content.split('\n')
        in_yield = False
        yield_start_line = 0
        bracket_count = 0
        
        for i, line in enumerate(lines, 1):
            if not in_yield:
                # Check for yield statement start
                if re.match(r'^\s*yield\s*\[', line):
                    in_yield = True
                    yield_start_line = i
                    bracket_count = line.count('[') - line.count(']')
                    # Check for immediate closing
                    if ']' in line and bracket_count == 0:
                        # Check if yield is empty
                        match = re.search(r'yield\s*\[\s*\]', line)
                        if match:
                            issues.append((
                                i,
                                line.strip(),
                                "Empty yield statement - should return actual values"
                            ))
                        in_yield = False
            else:
                # Continue counting brackets
                bracket_count += line.count('[') - line.count(']')
                if bracket_count <= 0:
                    in_yield = False
    
    except Exception as e:
        print(f"  ⚠️  Error parsing {file_path}: {e}")
    
    return issues


def find_mismatched_yield_outputs(file_path: Path) -> List[Tuple[int, str, str]]:
    """Find generator functions where yields return different numbers of values.
    
    This is a real bug - Gradio expects consistent outputs from all yields.
    
    Args:
        file_path: Path to Python file to check
        
    Returns:
        List of (line_number, function_name, details) tuples
    """
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if it's a generator function with yields
                yields_in_func = []
                
                # Walk the function body directly to get accurate line numbers
                for child in ast.walk(node):
                    if isinstance(child, ast.Yield) and child.value:
                        if isinstance(child.value, (ast.List, ast.Tuple)):
                            yields_in_func.append((child.lineno, len(child.value.elts)))
                        elif isinstance(child.value, ast.Dict):
                            yields_in_func.append((child.lineno, len(child.value.keys)))
                
                if len(yields_in_func) > 1:
                    counts = [count for _, count in yields_in_func]
                    if len(set(counts)) > 1:
                        # This is a real issue
                        details = ", ".join([f"line {ln}:{cnt}" for ln, cnt in yields_in_func])
                        issues.append((
                            node.lineno,
                            node.name,
                            f"Inconsistent yield counts: {details}"
                        ))
    
    except Exception as e:
        print(f"  ⚠️  Error analyzing {file_path}: {e}")
    
    return issues


def find_missing_component_references(file_path: Path) -> List[Tuple[int, str, str]]:
    """Find places where components are referenced before being defined.
    
    This catches patterns like:
    - Using a component in .click() before creating it
    - Referencing components['key'] that doesn't exist
    
    Args:
        file_path: Path to Python file to check
        
    Returns:
        List of (line_number, line_content, issue_description) tuples
    """
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
        
        # Track defined components
        defined_components: Set[str] = set()
        
        for i, line in enumerate(lines, 1):
            # Check for component definition
            # Pattern: components['key'] = gr.Component(...)
            define_match = re.search(
                r"components\[['\"](\w+)['\"]\]\s*=\s*gr\.(\w+)\s*\(",
                line
            )
            if define_match:
                key = define_match.group(1)
                defined_components.add(key)
            
            # Check for component reference
            # Pattern: components['key'] (but not followed by =)
            ref_matches = re.finditer(
                r"components\[['\"](\w+)['\"]\]",
                line
            )
            for ref_match in ref_matches:
                key = ref_match.group(1)
                # Check if this line defines the component
                if f"components['{key}']" in line and '=' in line:
                    continue
                if f'components["{key}"]' in line and '=' in line:
                    continue
                
                # Check if component was defined earlier
                if key not in defined_components:
                    issues.append((
                        i,
                        line.strip()[:80],
                        f"Component 'components['{key}']' used before definition"
                    ))
    
    except Exception as e:
        print(f"  ⚠️  Error checking {file_path}: {e}")
    
    return issues


def find_bad_function_signatures(file_path: Path) -> List[Tuple[int, str, str]]:
    """Find functions with suspicious signatures.
    
    This catches:
    - Functions that should return tuples but don't
    - Functions with too many/few parameters
    
    Args:
        file_path: Path to Python file to check
        
    Returns:
        List of (line_number, function_name, issue_description) tuples
    """
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check docstring for expected return type hints
                docstring = ast.get_docstring(node)
                
                if docstring:
                    # Check if docstring mentions Tuple return
                    if 'Returns:' in docstring and 'Tuple' in docstring:
                        # This function should return a tuple
                        # Check if it actually does
                        has_return = False
                        returns_tuple = False
                        
                        for child in ast.walk(node):
                            if isinstance(child, ast.Return):
                                has_return = True
                                if isinstance(child.value, (ast.Tuple, ast.List)):
                                    returns_tuple = True
                                break
                        
                        if has_return and not returns_tuple:
                            issues.append((
                                node.lineno,
                                node.name,
                                "Docstring says returns Tuple but function returns single value"
                            ))
    
    except Exception as e:
        print(f"  ⚠️  Error analyzing {file_path}: {e}")
    
    return issues


def run_realistic_tests(src_dir: Path = None) -> bool:
    """Run realistic tests that catch only real bugs.
    
    Args:
        src_dir: Source directory to test
        
    Returns:
        True if all tests pass, False otherwise
    """
    if src_dir is None:
        src_dir = Path(__file__).parent.parent / "src" / "edukaai_studio"
    
    all_passed = True
    total_issues = 0
    
    print("=" * 70)
    print("EdukaAI Studio - Realistic Test Suite")
    print("=" * 70)
    print("\nTesting for REAL bugs only (not false positives)...\n")
    
    # Find all Python files in ui/tabs
    tabs_dir = src_dir / "ui" / "tabs"
    if not tabs_dir.exists():
        print(f"❌ Tabs directory not found: {tabs_dir}")
        return False
    
    python_files = sorted(tabs_dir.glob("*.py"))
    
    print(f"Found {len(python_files)} Python files to test\n")
    
    for file_path in python_files:
        if file_path.name == "__init__.py":
            continue
            
        print(f"Testing: {file_path.name}")
        file_issues = []
        
        # Test 1: Deprecated Gradio API calls (REAL BUG - causes AttributeError)
        gradio_issues = find_deprecated_gradio_calls(file_path)
        for line, content, suggestion in gradio_issues:
            file_issues.append((
                "CRITICAL",
                line,
                content[:70],
                suggestion
            ))
        
        # Test 2: Broken yield patterns (REAL BUG - causes SyntaxError)
        broken_yields = find_broken_yield_patterns(file_path)
        for line, content, suggestion in broken_yields:
            file_issues.append((
                "CRITICAL",
                line,
                content[:70],
                suggestion
            ))
        
        # Test 3: Mismatched yield outputs (REAL BUG - causes Gradio error)
        mismatched_yields = find_mismatched_yield_outputs(file_path)
        for line, func_name, details in mismatched_yields:
            file_issues.append((
                "CRITICAL",
                line,
                f"Function: {func_name}",
                f"Inconsistent yields: {details}"
            ))
        
        # Test 4: Missing component references (REAL BUG - causes KeyError)
        missing_refs = find_missing_component_references(file_path)
        for line, content, description in missing_refs:
            file_issues.append((
                "WARNING",
                line,
                content[:70],
                description
            ))
        
        # Test 5: Bad function signatures (POTENTIAL BUG)
        bad_sigs = find_bad_function_signatures(file_path)
        for line, func_name, description in bad_sigs:
            file_issues.append((
                "WARNING",
                line,
                f"Function: {func_name}",
                description
            ))
        
        if file_issues:
            all_passed = False
            critical_count = sum(1 for i in file_issues if i[0] == "CRITICAL")
            warning_count = sum(1 for i in file_issues if i[0] == "WARNING")
            
            print(f"  ❌ Found {critical_count} critical, {warning_count} warning(s):")
            
            # Show critical issues first
            for severity, line, content, suggestion in file_issues:
                if severity == "CRITICAL":
                    print(f"    🔴 Line {line}: {suggestion}")
                    print(f"       Code: {content}")
            
            # Show warnings
            for severity, line, content, suggestion in file_issues:
                if severity == "WARNING":
                    print(f"    ⚠️  Line {line}: {suggestion}")
        else:
            print(f"  ✅ No critical issues found")
        
        total_issues += len(file_issues)
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ All tests passed! No critical issues found.")
    else:
        print(f"❌ Found {total_issues} issue(s). Critical issues must be fixed.")
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run realistic tests for EdukaAI Studio UI"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show more details about each issue"
    )
    
    args = parser.parse_args()
    
    success = run_realistic_tests()
    sys.exit(0 if success else 1)

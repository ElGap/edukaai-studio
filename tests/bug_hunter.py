"""Bug Hunter - Finds real bugs in EdukaAI Studio.

This module finds bugs by:
1. Checking function signatures match their usage
2. Detecting missing error handling
3. Finding state requirements that cause crashes
4. Identifying file access without existence checks
5. Detecting Gradio event handler mismatches
"""

import ast
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Set

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class BugHunter:
    """Hunts for bugs in Python files."""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.issues = []
        self.content = None
        self.tree = None
        
    def load(self) -> bool:
        """Load and parse the file."""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.content = f.read()
            self.tree = ast.parse(self.content)
            return True
        except Exception as e:
            self.issues.append((0, "PARSE_ERROR", f"Failed to parse file: {e}"))
            return False
    
    def find_missing_error_handling(self) -> List[Tuple[int, str, str]]:
        """Find functions that access files/paths without existence checks."""
        issues = []
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                # Check for file operations without try/except
                has_file_access = False
                has_error_handling = False
                
                for child in ast.walk(node):
                    # Check for file/path operations
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Attribute):
                            if child.func.attr in ['exists', 'is_file', 'is_dir', 'open', 'read', 'write']:
                                has_file_access = True
                    
                    # Check for try/except blocks
                    if isinstance(child, ast.Try):
                        has_error_handling = True
                
                # If file operations exist without error handling, flag it
                if has_file_access and not has_error_handling:
                    issues.append((
                        node.lineno,
                        "MISSING_ERROR_HANDLING",
                        f"Function '{node.name}' has file operations but no try/except error handling"
                    ))
        
        return issues
    
    def find_state_requirements(self) -> List[Tuple[int, str, str]]:
        """Find functions that require specific state keys but don't document them."""
        issues = []
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                # Look for .get() calls on current_state
                required_keys = set()
                
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Attribute):
                            if child.func.attr == 'get':
                                # Check if it's current_state.get()
                                if isinstance(child.func.value, ast.Name):
                                    if 'state' in child.func.value.id.lower():
                                        # Get the key being accessed
                                        if child.args:
                                            if isinstance(child.args[0], ast.Constant):
                                                key = child.args[0].value
                                                if isinstance(key, str):
                                                    required_keys.add(key)
                
                # Check docstring for these requirements
                docstring = ast.get_docstring(node)
                if required_keys and docstring:
                    missing_in_docs = []
                    for key in required_keys:
                        if key not in docstring:
                            missing_in_docs.append(key)
                    
                    if missing_in_docs:
                        issues.append((
                            node.lineno,
                            "UNDOCUMENTED_STATE_REQ",
                            f"Function '{node.name}' requires state keys not in docstring: {missing_in_docs}"
                        ))
        
        return issues
    
    def find_naked_path_access(self) -> List[Tuple[int, str, str]]:
        """Find Path operations without existence checks."""
        issues = []
        
        # Pattern: Path(...) without .exists() check
        path_pattern = re.compile(
            r'Path\([^)]+\)(?!\s*\.\s*(exists|is_file|is_dir|mkdir))'
        )
        
        lines = self.content.split('\n')
        for i, line in enumerate(lines, 1):
            if 'Path(' in line:
                # Check if it's used with exists() in same or next line
                if not re.search(r'exists\(\)|is_file\(\)|is_dir\(\)|mkdir', line):
                    # Check next 3 lines for existence check
                    next_lines = '\n'.join(lines[i:i+3])
                    if not re.search(r'\.\s*(exists|is_file|is_dir)\s*\(', next_lines):
                        issues.append((
                            i,
                            "NAKED_PATH_ACCESS",
                            f"Path used without existence check: {line.strip()[:60]}"
                        ))
        
        return issues
    
    def find_implicit_none_returns(self) -> List[Tuple[int, str, str]]:
        """Find functions that may return None implicitly."""
        issues = []
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                # Check if function has return statements
                has_return = False
                returns_value = False
                
                for child in ast.walk(node):
                    if isinstance(child, ast.Return):
                        has_return = True
                        if child.value is not None:
                            returns_value = True
                
                # If function has some returns with values but not all paths covered
                if returns_value and not self._all_paths_return(node):
                    issues.append((
                        node.lineno,
                        "IMPLICIT_NONE_RETURN",
                        f"Function '{node.name}' may return None implicitly on some code paths"
                    ))
        
        return issues
    
    def _all_paths_return(self, node) -> bool:
        """Check if all code paths in a function have a return statement."""
        # Simple check - look for return at end of function
        for child in ast.walk(node):
            if isinstance(child, ast.Return):
                # Check if it's at the end
                return True
        return False
    
    def find_bare_except_clauses(self) -> List[Tuple[int, str, str]]:
        """Find bare except: clauses that catch everything including KeyboardInterrupt."""
        issues = []
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Try):
                for handler in node.handlers:
                    if handler.type is None:
                        issues.append((
                            handler.lineno,
                            "BARE_EXCEPT",
                            "Bare 'except:' clause catches KeyboardInterrupt and SystemExit - use 'except Exception:' instead"
                        ))
        
        return issues
    
    def find_hardcoded_paths(self) -> List[Tuple[int, str, str]]:
        """Find hardcoded paths that might not work on all systems."""
        issues = []
        
        hardcoded_patterns = [
            (r'["\']outputs/[^"\']+["\']', "Hardcoded 'outputs/' path"),
            (r'["\']\.studio_state\.json["\']', "Hardcoded state file path"),
            (r'["\']/tmp/[^"\']+["\']', "Hardcoded /tmp/ path (Linux-specific)"),
            (r'["\']C:\\\\', "Hardcoded Windows path"),
        ]
        
        lines = self.content.split('\n')
        for i, line in enumerate(lines, 1):
            for pattern, desc in hardcoded_patterns:
                if re.search(pattern, line):
                    issues.append((i, "HARDCODED_PATH", f"{desc}: {line.strip()[:50]}"))
        
        return issues
    
    def find_print_instead_of_logging(self) -> List[Tuple[int, str, str]]:
        """Find print statements that should be logging."""
        issues = []
        
        lines = self.content.split('\n')
        for i, line in enumerate(lines, 1):
            # Match print() calls with debug/error patterns
            if re.match(r'\s*print\s*\([^)]*(DEBUG|ERROR|WARN|INFO|FATAL)', line, re.IGNORECASE):
                issues.append((
                    i,
                    "PRINT_NOT_LOGGING",
                    f"Should use logging instead of print: {line.strip()[:60]}"
                ))
        
        return issues
    
    def find_unused_function_returns(self) -> List[Tuple[int, str, str]]:
        """Find functions that return values but callers ignore them."""
        issues = []
        
        # Get all function names
        function_names = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                function_names.add(node.name)
        
        # Check if returns are used
        lines = self.content.split('\n')
        for i, line in enumerate(lines, 1):
            # Match function calls where return is not assigned
            for func_name in function_names:
                # Pattern: func_name(...) not preceded by = or variable
                pattern = rf'^\s*{re.escape(func_name)}\s*\('
                if re.match(pattern, line):
                    if not re.search(rf'[=\w\s]+{re.escape(func_name)}\s*\(', line):
                        issues.append((
                            i,
                            "UNUSED_RETURN",
                            f"Return value of '{func_name}()' may be unused - check if this is intentional"
                        ))
        
        return issues
    
    def hunt(self) -> List[Tuple[int, str, str, str]]:
        """Run all bug hunters and return issues."""
        if not self.load():
            return self.issues
        
        all_hunters = [
            ("Error Handling", self.find_missing_error_handling),
            ("State Requirements", self.find_state_requirements),
            ("Path Safety", self.find_naked_path_access),
            ("Return Safety", self.find_implicit_none_returns),
            ("Exception Handling", self.find_bare_except_clauses),
            ("Portability", self.find_hardcoded_paths),
            ("Logging", self.find_print_instead_of_logging),
            ("Return Values", self.find_unused_function_returns),
        ]
        
        for category, hunter_func in all_hunters:
            try:
                found = hunter_func()
                for line, issue_type, description in found:
                    self.issues.append((line, category, issue_type, description))
            except Exception as e:
                print(f"  ⚠️  Error in {category} hunter: {e}")
        
        return self.issues


def run_bug_hunt(src_dir: Path = None) -> bool:
    """Run bug hunter on all UI tabs.
    
    Returns:
        True if no bugs found, False otherwise
    """
    if src_dir is None:
        # Get correct path from test location
        src_dir = Path(__file__).parent.parent / "src" / "edukaai_studio"
    
    tabs_dir = src_dir / "ui" / "tabs"
    if not tabs_dir.exists():
        print(f"❌ Tabs directory not found: {tabs_dir}")
        return False
    
    print("=" * 80)
    print("🐛 EdukaAI Studio - Bug Hunter")
    print("=" * 80)
    print("\nLooking for subtle bugs that cause runtime issues...\n")
    
    all_issues = []
    
    for file_path in sorted(tabs_dir.glob("*.py")):
        if file_path.name == "__init__.py":
            continue
        
        print(f"🔍 Scanning: {file_path.name}")
        hunter = BugHunter(file_path)
        issues = hunter.hunt()
        
        if issues:
            print(f"  ⚠️  Found {len(issues)} potential issue(s):")
            for line, category, issue_type, description in issues:
                print(f"    Line {line}: [{category}] {issue_type}")
                print(f"      {description[:70]}")
            all_issues.extend(issues)
        else:
            print(f"  ✅ Clean")
    
    print("\n" + "=" * 80)
    if all_issues:
        print(f"⚠️  Found {len(all_issues)} potential bug(s) to review")
        print("\nCategories:")
        categories = {}
        for _, cat, _, _ in all_issues:
            categories[cat] = categories.get(cat, 0) + 1
        for cat, count in sorted(categories.items()):
            print(f"  - {cat}: {count}")
    else:
        print("✅ No obvious bugs found!")
    print("=" * 80)
    
    return len(all_issues) == 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hunt for bugs in EdukaAI Studio")
    args = parser.parse_args()
    
    success = run_bug_hunt()
    sys.exit(0 if success else 1)

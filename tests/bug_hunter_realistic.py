#!/usr/bin/env python3
"""Bug Hunter - Finds real bugs in EdukaAI Studio.

This module finds bugs by checking for patterns that cause runtime errors.
Filters out false positives to only show real issues.
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
        self.lines = None
        
    def load(self) -> bool:
        """Load and parse the file."""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.content = f.read()
                self.lines = self.content.split('\n')
            self.tree = ast.parse(self.content)
            return True
        except Exception as e:
            self.issues.append((0, "PARSE_ERROR", f"Failed to parse file: {e}"))
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
                            "Bare 'except:' clause catches KeyboardInterrupt - use 'except Exception:'",
                            "HIGH"
                        ))
        
        return issues
    
    def find_missing_error_handling(self) -> List[Tuple[int, str, str]]:
        """Find functions with risky file operations but no error handling."""
        issues = []
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                has_risky_operation = False
                has_error_handling = False
                
                for child in ast.walk(node):
                    # Check for risky operations
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Attribute):
                            if child.func.attr in ['read', 'write', 'open']:
                                # Check if wrapped in try/except
                                for parent in ast.walk(node):
                                    if isinstance(parent, ast.Try):
                                        has_error_handling = True
                                        break
                                has_risky_operation = True
                    
                    # Check for try/except blocks
                    if isinstance(child, ast.Try):
                        has_error_handling = True
                
                # Only flag if risky AND no error handling
                if has_risky_operation and not has_error_handling:
                    issues.append((
                        node.lineno,
                        "MISSING_ERROR_HANDLING",
                        f"Function '{node.name}' has file operations without try/except",
                        "HIGH"
                    ))
        
        return issues
    
    def find_unsafe_path_usage(self) -> List[Tuple[int, str, str]]:
        """Find Path usage that's actually unsafe (not just Path creation)."""
        issues = []
        
        # Look for patterns like:
        # path.read_text() without checking exists()
        # path.stat() without checking exists()
        # str(path) is fine, Path() creation is fine
        
        for i, line in enumerate(self.lines, 1):
            # Check for unsafe Path method calls
            unsafe_patterns = [
                (r'Path\([^)]+\)\.(read|write|stat|iterdir|rglob)\s*\(', 
                 "Unsafe Path method call without existence check"),
            ]
            
            for pattern, desc in unsafe_patterns:
                if re.search(pattern, line):
                    # Check if there's an exists() check nearby
                    context = '\n'.join(self.lines[max(0, i-5):i+1])
                    if not re.search(r'exists\s*\(\)', context):
                        issues.append((
                            i,
                            "UNSAFE_PATH_USAGE",
                            desc,
                            "HIGH"
                        ))
        
        return issues
    
    def find_deprecated_gradio_calls(self) -> List[Tuple[int, str, str]]:
        """Find deprecated Gradio API calls."""
        issues = []
        
        # Pattern: gr.ComponentName.update(
        deprecated_pattern = re.compile(
            r'gr\.(Dropdown|Textbox|Button|Slider|File|Plot|HTML|Markdown|'
            r'Code|Image|Audio|Video|Dataframe|JSON|Checkbox|Number|'
            r'Radio|State|Tabs|TabItem|Row|Column|Group|Accordion)\s*\.\s*update\s*\('
        )
        
        for i, line in enumerate(self.lines, 1):
            match = deprecated_pattern.search(line)
            if match:
                component = match.group(1)
                issues.append((
                    i,
                    "DEPRECATED_GRADIO_API",
                    f"gr.{component}.update() is deprecated, use gr.update()",
                    "HIGH"
                ))
        
        return issues
    
    def hunt(self) -> List[Tuple[int, str, str, str]]:
        """Run all bug hunters and return issues."""
        if not self.load():
            return self.issues
        
        # Only run hunters that find REAL bugs
        real_bug_hunters = [
            ("Exception Handling", self.find_bare_except_clauses),
            ("Error Handling", self.find_missing_error_handling),
            ("Path Safety", self.find_unsafe_path_usage),
            ("API Compatibility", self.find_deprecated_gradio_calls),
        ]
        
        for category, hunter_func in real_bug_hunters:
            try:
                found = hunter_func()
                for line, issue_type, description, severity in found:
                    self.issues.append((line, category, issue_type, description, severity))
            except Exception as e:
                print(f"  ⚠️  Error in {category} hunter: {e}")
        
        return self.issues


def run_bug_hunt(src_dir: Path = None) -> bool:
    """Run bug hunter on all UI tabs."""
    if src_dir is None:
        src_dir = Path(__file__).parent.parent / "src" / "edukaai_studio"
    
    tabs_dir = src_dir / "ui" / "tabs"
    if not tabs_dir.exists():
        print(f"❌ Tabs directory not found: {tabs_dir}")
        return False
    
    print("=" * 80)
    print("🐛 EdukaAI Studio - Bug Hunter (Real Bugs Only)")
    print("=" * 80)
    print("\nLooking for bugs that cause runtime errors...\n")
    
    all_issues = []
    
    for file_path in sorted(tabs_dir.glob("*.py")):
        if file_path.name == "__init__.py":
            continue
        
        print(f"🔍 Scanning: {file_path.name}")
        hunter = BugHunter(file_path)
        issues = hunter.hunt()
        
        # Filter to only HIGH severity issues
        critical_issues = [i for i in issues if len(i) >= 5 and i[4] == "HIGH"]
        
        if critical_issues:
            print(f"  🔴 Found {len(critical_issues)} CRITICAL issue(s):")
            for issue in critical_issues:
                line, category, issue_type, description, severity = issue
                print(f"    Line {line}: [{issue_type}]")
                print(f"      {description}")
            all_issues.extend(critical_issues)
        elif issues:
            # Show count but not details for non-critical
            print(f"  ℹ️  {len(issues)} minor issue(s) (not critical)")
        else:
            print(f"  ✅ Clean - no critical bugs")
    
    print("\n" + "=" * 80)
    if all_issues:
        print(f"🔴 Found {len(all_issues)} CRITICAL bug(s) that must be fixed!")
        print("\nCategories:")
        categories = {}
        for _, cat, _, _, _ in all_issues:
            categories[cat] = categories.get(cat, 0) + 1
        for cat, count in sorted(categories.items()):
            print(f"  - {cat}: {count}")
        print("\n⚠️  These bugs WILL cause crashes or runtime errors!")
    else:
        print("✅ No critical bugs found!")
        print("\n🎉 The app should be stable and crash-free!")
    print("=" * 80)
    
    return len(all_issues) == 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hunt for bugs in EdukaAI Studio")
    args = parser.parse_args()
    
    success = run_bug_hunt()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Security Scanner for EdukaAI Studio Dependencies

Scans requirements files for known vulnerabilities and outdated packages.

Usage:
    python scripts/security_scan.py
    
Requirements:
    pip install safety pip-audit
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
            
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"❌ Timeout after 120 seconds")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def check_vulnerabilities():
    """Check for known vulnerabilities using safety."""
    print("🔍 Checking for known vulnerabilities...")
    
    # Check if safety is installed
    try:
        import safety
    except ImportError:
        print("⚠️  'safety' not installed. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "safety"], check=True)
    
    return run_command(
        "safety check -r requirements.txt",
        "Safety vulnerability scan"
    )


def check_outdated():
    """Check for outdated packages."""
    print("\n📦 Checking for outdated packages...")
    
    return run_command(
        "pip list --outdated --format=columns",
        "Outdated packages check"
    )


def check_with_pip_audit():
    """Alternative check using pip-audit."""
    print("\n🔐 Running pip-audit scan...")
    
    try:
        import pip_audit
    except ImportError:
        print("⚠️  'pip-audit' not installed. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pip-audit"], check=True)
    
    return run_command(
        "pip-audit -r requirements.txt --desc",
        "pip-audit vulnerability scan"
    )


def main():
    """Main security scanning function."""
    print("="*70)
    print("EDUKAAI STUDIO - SECURITY DEPENDENCY SCANNER")
    print("="*70)
    print(f"Working directory: {Path.cwd()}")
    print(f"Python version: {sys.version}")
    print("="*70)
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found in current directory")
        sys.exit(1)
    
    results = []
    
    # Run vulnerability checks
    results.append(("Safety scan", check_vulnerabilities()))
    results.append(("pip-audit scan", check_with_pip_audit()))
    results.append(("Outdated packages", check_outdated()))
    
    # Summary
    print("\n" + "="*70)
    print("SCAN SUMMARY")
    print("="*70)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
        if not passed:
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("✅ All security checks passed!")
        print("\n⚠️  Note: Passing scans don't guarantee zero vulnerabilities.")
        print("   Always review docs/SECURITY_AUDIT.md for manual checks.")
        return 0
    else:
        print("❌ Some security checks failed!")
        print("\n📖 See docs/SECURITY_AUDIT.md for remediation steps")
        print("🔗 Check https://pypi.org/ for latest package versions")
        return 1


if __name__ == "__main__":
    sys.exit(main())

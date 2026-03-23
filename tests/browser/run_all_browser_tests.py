"""Comprehensive browser test suite runner.

Runs all browser tests and generates a detailed report.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json


def run_browser_test_suite():
    """Run all browser tests and report results."""
    
    print("=" * 80)
    print("🎭 EdukaAI Studio - Comprehensive Browser Test Suite")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test files to run
    test_files = [
        ("Upload Tab", "tests/browser/test_upload.py"),
        ("Configure Tab", "tests/browser/test_configure.py"),
        ("Train Tab", "tests/browser/test_train.py"),
        ("Results Tab", "tests/browser/test_results.py"),
        ("Chat Tab", "tests/browser/test_chat.py"),
        ("My Models Tab", "tests/browser/test_my_models.py"),
        ("Models Tab", "tests/browser/test_models.py"),
        ("Workflows", "tests/browser/test_workflow.py"),
        ("Error Scenarios", "tests/browser/test_errors.py"),
        ("State Persistence", "tests/browser/test_state.py"),
    ]
    
    results = {}
    total_passed = 0
    total_failed = 0
    total_tests = 0
    
    project_root = Path(__file__).parent.parent
    
    # Check if app is running
    print("Checking if app is running...")
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:7860", timeout=2)
        print("✅ App is running at http://localhost:7860")
    except:
        print("⚠️  App not detected. Starting app...")
        print("Please run: python src/edukaai_studio/main_simplified.py &")
        print("Then wait 10 seconds and run this script again.")
        return False
    
    print()
    
    for test_name, test_file in test_files:
        print(f"\n{'='*80}")
        print(f"Running: {test_name}")
        print(f"File: {test_file}")
        print('='*80)
        
        test_path = project_root / test_file
        if not test_path.exists():
            print(f"❌ Test file not found: {test_file}")
            results[test_name] = {"status": "NOT_FOUND", "tests": 0, "passed": 0, "failed": 0}
            continue
        
        try:
            # Run the test file
            result = subprocess.run(
                ["python", "-m", "pytest", str(test_path), "--headed=false", "-v", "--tb=short"],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per test file
            )
            
            # Parse output
            output = result.stdout + result.stderr
            
            # Count passes and fails
            import re
            passed_count = len(re.findall(r'PASSED', output))
            failed_count = len(re.findall(r'FAILED', output))
            error_count = len(re.findall(r'ERROR', output))
            
            total_tests_in_file = passed_count + failed_count + error_count
            
            results[test_name] = {
                "status": "PASS" if result.returncode == 0 else "FAIL",
                "tests": total_tests_in_file,
                "passed": passed_count,
                "failed": failed_count + error_count,
                "output": output[-2000:] if len(output) > 2000 else output  # Last 2000 chars
            }
            
            total_tests += total_tests_in_file
            total_passed += passed_count
            total_failed += failed_count + error_count
            
            if result.returncode == 0:
                print(f"✅ {test_name}: {passed_count}/{total_tests_in_file} passed")
            else:
                print(f"❌ {test_name}: {failed_count} failed")
                if failed_count > 0:
                    # Show failed test names
                    failed_tests = re.findall(r'FAILED\s+(\S+)', output)
                    for ft in failed_tests[:3]:  # Show first 3
                        print(f"   - {ft}")
            
        except subprocess.TimeoutExpired:
            print(f"⏱️  {test_name}: Timeout (5 minutes)")
            results[test_name] = {"status": "TIMEOUT", "tests": 0, "passed": 0, "failed": 0}
        except Exception as e:
            print(f"❌ {test_name}: Error - {e}")
            results[test_name] = {"status": "ERROR", "tests": 0, "passed": 0, "failed": 0}
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal Tests: {total_tests}")
    print(f"Passed: {total_passed} ({total_passed/max(total_tests,1)*100:.1f}%)")
    print(f"Failed: {total_failed} ({total_failed/max(total_tests,1)*100:.1f}%)")
    
    print("\nBy Component:")
    for test_name, result in results.items():
        status_emoji = "✅" if result["status"] == "PASS" else "❌" if result["status"] == "FAIL" else "⚠️"
        print(f"{status_emoji} {test_name}: {result['passed']}/{result['tests']} passed")
    
    print("\n" + "=" * 80)
    if total_failed == 0 and all(r["status"] == "PASS" for r in results.values()):
        print("🎉 ALL TESTS PASSED!")
        print("The UI is fully functional and ready for use.")
    else:
        print(f"⚠️  {total_failed} test(s) failed")
        print("Review the output above for details.")
    print("=" * 80)
    
    # Save detailed report
    report_file = project_root / "BROWSER_TEST_REPORT.json"
    with open(report_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "pass_rate": total_passed/max(total_tests,1)*100,
            "results": results
        }, f, indent=2)
    
    print(f"\nDetailed report saved to: {report_file}")
    
    return total_failed == 0


if __name__ == "__main__":
    success = run_browser_test_suite()
    sys.exit(0 if success else 1)

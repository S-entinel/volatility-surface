#!/usr/bin/env python3
"""
Test runner script for volatility surface analyzer.

Runs all tests with coverage reporting and generates HTML reports.
"""

import sys
import subprocess
from pathlib import Path


def run_tests():
    """Run all tests with pytest."""
    print("=" * 80)
    print("Running Volatility Surface Analyzer Test Suite")
    print("=" * 80)
    print()
    
    # Run pytest with coverage
    cmd = [
        "pytest",
        "-v",                       # Verbose
        "--tb=short",               # Short traceback
        "--cov=src",                # Coverage for src
        "--cov-report=html",        # HTML report
        "--cov-report=term-missing", # Terminal with missing lines
        "-ra",                      # Summary of all outcomes
    ]
    
    result = subprocess.run(cmd)
    
    print()
    print("=" * 80)
    if result.returncode == 0:
        print("✅ All tests passed!")
        print("📊 Coverage report: htmlcov/index.html")
    else:
        print("❌ Some tests failed")
        print(f"   Exit code: {result.returncode}")
    print("=" * 80)
    
    return result.returncode


def run_quick_tests():
    """Run only unit tests (fast)."""
    print("Running quick unit tests...")
    cmd = [
        "pytest",
        "-v",
        "-m", "unit",
        "--tb=short",
    ]
    
    result = subprocess.run(cmd)
    return result.returncode


def run_integration_tests():
    """Run integration tests."""
    print("Running integration tests...")
    cmd = [
        "pytest",
        "-v",
        "-m", "integration",
        "--tb=short",
    ]
    
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            sys.exit(run_quick_tests())
        elif sys.argv[1] == "integration":
            sys.exit(run_integration_tests())
    
    sys.exit(run_tests())
#!/usr/bin/env python3
"""
Script to run performance tests and generate reports
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def run_performance_tests(verbose=False, report_file=None):
    """Run performance tests and collect results"""
    
    # Base command
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/performance/", 
        "-m", "performance",
        "-v" if verbose else "-q",
        "--tb=short"
    ]
    
    print("🚀 Running Performance Tests...")
    print(f"Command: {' '.join(cmd)}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        success = True
        output = result.stdout
        error_output = result.stderr
    except subprocess.CalledProcessError as e:
        success = False
        output = e.stdout
        error_output = e.stderr
    
    duration = time.time() - start_time
    
    print(output)
    if error_output:
        print("STDERR:")
        print(error_output)
    
    # Generate report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "success": success,
        "duration_seconds": round(duration, 2),
        "command": " ".join(cmd),
        "output": output,
        "error_output": error_output
    }
    
    if report_file:
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📊 Report saved to: {report_file}")
    
    print(f"\n⏱️  Total duration: {duration:.2f} seconds")
    print(f"{'✅ PASSED' if success else '❌ FAILED'}")
    
    return success


def main():
    parser = argparse.ArgumentParser(description="Run structured logging performance tests")
    parser.add_argument("-v", "--verbose", action="store_true", 
                       help="Verbose output")
    parser.add_argument("-r", "--report", type=str, 
                       help="Save report to JSON file")
    parser.add_argument("--baseline", action="store_true",
                       help="Run only baseline performance tests")
    
    args = parser.parse_args()
    
    if args.baseline:
        # Run only baseline tests
        cmd = [
            sys.executable, "-m", "pytest", 
            "tests/performance/test_benchmarks.py::TestPerformanceRegression::test_baseline_performance_requirements",
            "-v", "-s"
        ]
        print("🎯 Running Baseline Performance Tests...")
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    
    success = run_performance_tests(args.verbose, args.report)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
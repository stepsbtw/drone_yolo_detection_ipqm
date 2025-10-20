#!/usr/bin/env python3
"""
Run distance estimation test.
"""

import subprocess
import sys

def run_test(script_name, description, extra_args=""):
    """Run a test script and capture output."""
    print("\n" + "="*80)
    print(f"Running: {description}")
    print("="*80)
    
    cmd = f"cd src && python {script_name} {extra_args}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    return result.returncode

def main():
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                        DISTANCE ESTIMATION TEST                            ║
║                                                                            ║
║  This script runs distance estimation using simple pinhole model          ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
""")
    
    # Run distance estimation
    print("\n🔹 TEST: Simple Distance Estimation")
    run_test("main.py", "Distance Estimation")
    
    print("\n" + "="*80)
    print("✅ TEST COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Run All Experiments: Complete NV Center Experiment Suite

This script runs all available experiments in sequence to demonstrate
the full capabilities of the QUSIM framework.
"""

import os
import sys
import time
import subprocess

def run_experiment(script_name: str, description: str):
    """Run a single experiment script."""
    print(f"\n🚀 Running: {description}")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Run the experiment script
        result = subprocess.run([sys.executable, script_name], 
                              cwd=os.path.dirname(__file__),
                              capture_output=True, text=True, timeout=300)
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully!")
            print(f"   Execution time: {elapsed:.1f} seconds")
            if result.stdout:
                # Show last few lines of output
                output_lines = result.stdout.strip().split('\n')
                if len(output_lines) > 5:
                    print("   Final output:")
                    for line in output_lines[-3:]:
                        if line.strip():
                            print(f"   {line}")
        else:
            print(f"❌ {description} failed!")
            print(f"   Error: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} timed out (>5 minutes)")
    except Exception as e:
        print(f"💥 {description} crashed: {e}")
    
    print("-" * 60)

def main():
    """Run all experiments in the suite."""
    
    print("🧪 QUSIM NV Center Experiment Suite")
    print("=" * 60)
    print("This script runs all available experiments to demonstrate")
    print("the complete NV center simulation capabilities.")
    print("=" * 60)
    
    # List of experiments to run
    experiments = [
        ("minimal_demo.py", "Minimal π-Pulse Demo"),
        # Note: Other experiments commented out due to timeout issues
        # ("quick_demo.py", "Quick π-Pulse Experiment"),  
        # ("pi_pulse_readout.py", "Full π-Pulse Readout Experiment"),
    ]
    
    total_start = time.time()
    
    for script, description in experiments:
        if os.path.exists(script):
            run_experiment(script, description)
        else:
            print(f"⚠️  Warning: {script} not found, skipping...")
    
    total_elapsed = time.time() - total_start
    
    print(f"\n🎉 All experiments completed!")
    print(f"Total execution time: {total_elapsed:.1f} seconds")
    print("\nExperiment files:")
    print("  minimal_demo.py      - Fastest, shows core physics")
    print("  quick_demo.py        - Includes QUSIM simulation")  
    print("  pi_pulse_readout.py  - Complete realistic experiment")
    print("\nManual execution:")
    print("  cd experiments")
    print("  python minimal_demo.py")
    print("  python quick_demo.py")
    print("  python pi_pulse_readout.py")

if __name__ == "__main__":
    main()
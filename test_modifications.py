#!/usr/bin/env python3
"""
Quick test script to verify the modifications work correctly.
"""

import sys
import os
import subprocess

def test_argument_parsing():
    """Test if argument parsing works in both scripts."""
    print("🧪 Testing argument parsing...")
    
    # Test train_tiktok.py
    try:
        result = subprocess.run([
            sys.executable, 'train_tiktok.py', '--help'
        ], capture_output=True, text=True, timeout=10)
        
        if '--rd_state' in result.stdout:
            print("✅ train_tiktok.py: Argument parsing works")
        else:
            print("❌ train_tiktok.py: --rd_state not found in help")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
    except subprocess.TimeoutExpired:
        print("⚠️ train_tiktok.py: Help command timed out (might be loading models)")
    except Exception as e:
        print(f"❌ train_tiktok.py: Error - {e}")
    
    # Test test_tiktok.py
    try:
        result = subprocess.run([
            sys.executable, 'test_tiktok.py', '--help'
        ], capture_output=True, text=True, timeout=10)
        
        if '--rd_state' in result.stdout and '--save_predictions' in result.stdout:
            print("✅ test_tiktok.py: Argument parsing works")
        else:
            print("❌ test_tiktok.py: Arguments not found in help")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
    except subprocess.TimeoutExpired:
        print("⚠️ test_tiktok.py: Help command timed out (might be loading models)")
    except Exception as e:
        print(f"❌ test_tiktok.py: Error - {e}")

def test_syntax():
    """Test if the Python files have valid syntax."""
    print("\n🔍 Testing Python syntax...")
    
    files_to_check = ['train_tiktok.py', 'test_tiktok.py', 'run_experiments.py']
    
    for filename in files_to_check:
        try:
            with open(filename, 'r') as f:
                content = f.read()
            
            # Try to compile the code
            compile(content, filename, 'exec')
            print(f"✅ {filename}: Syntax is valid")
            
        except SyntaxError as e:
            print(f"❌ {filename}: Syntax error at line {e.lineno}: {e.msg}")
        except FileNotFoundError:
            print(f"⚠️ {filename}: File not found")
        except Exception as e:
            print(f"❌ {filename}: Error - {e}")

def show_usage_examples():
    """Show usage examples."""
    print("\n📚 USAGE EXAMPLES:")
    print("="*50)
    print("# Train with different random states:")
    print("python train_tiktok.py --rd_state 1")
    print("python train_tiktok.py --rd_state 42")
    print("python train_tiktok.py --rd_state 100")
    print()
    print("# Test with detailed results:")
    print("python test_tiktok.py --rd_state 1 --save_predictions")
    print("python test_tiktok.py --rd_state 42 --save_predictions")
    print()
    print("# Run multiple experiments:")
    print("python run_experiments.py")
    print()
    print("# Check help:")
    print("python train_tiktok.py --help")
    print("python test_tiktok.py --help")

def main():
    print("🚀 TESTING MODIFIED SCRIPTS")
    print("="*40)
    
    # Change to the correct directory
    os.chdir('/workspace/ToxVidLM_ACL_2024')
    
    # Run tests
    test_syntax()
    test_argument_parsing()
    show_usage_examples()
    
    print(f"\n{'='*40}")
    print("✨ Test complete! Check the results above.")
    print("If syntax is valid, you can start using the new functionality!")

if __name__ == "__main__":
    main()

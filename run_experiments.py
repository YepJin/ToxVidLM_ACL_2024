#!/usr/bin/env python3
"""
Script to run experiments with different random states and collect results.
This demonstrates how to use the modified train_tiktok.py and test_tiktok.py files.
"""

import subprocess
import json
import os
import pandas as pd
from datetime import datetime

def run_experiment(rd_state):
    """Run training and testing with a specific random state."""
    print(f"\n{'='*60}")
    print(f"🚀 RUNNING EXPERIMENT WITH RANDOM STATE: {rd_state}")
    print(f"{'='*60}")
    
    try:
        # Train the model
        print(f"📚 Training model with rd_state={rd_state}...")
        train_cmd = ["python", "train_tiktok.py", "--rd_state", str(rd_state)]
        train_result = subprocess.run(train_cmd, capture_output=True, text=True, timeout=3600)
        
        if train_result.returncode != 0:
            print(f"❌ Training failed for rd_state={rd_state}")
            print(f"Error: {train_result.stderr}")
            return None
        
        print(f"✅ Training completed for rd_state={rd_state}")
        
        # Test the model and save detailed results
        print(f"🧪 Testing model with rd_state={rd_state}...")
        test_cmd = ["python", "test_tiktok.py", "--rd_state", str(rd_state), "--save_predictions"]
        test_result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=1800)
        
        if test_result.returncode != 0:
            print(f"❌ Testing failed for rd_state={rd_state}")
            print(f"Error: {test_result.stderr}")
            return None
        
        print(f"✅ Testing completed for rd_state={rd_state}")
        
        # Load and return the results
        results_file = f"results/test_results_rd{rd_state}.json"
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            return results
        else:
            print(f"⚠️ Results file not found: {results_file}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Experiment timed out for rd_state={rd_state}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error for rd_state={rd_state}: {e}")
        return None

def compare_results(results_list):
    """Compare results from different random states."""
    if not results_list:
        print("No results to compare!")
        return
    
    print(f"\n{'='*60}")
    print("📊 COMPARING RESULTS ACROSS DIFFERENT RANDOM STATES")
    print(f"{'='*60}")
    
    # Create comparison table
    comparison_data = []
    
    for result in results_list:
        rd_state = result["random_state"]
        for task in result["test_metrics"]:
            metrics = result["test_metrics"][task]
            detailed = result["detailed_analysis"][task]
            
            comparison_data.append({
                "Random_State": rd_state,
                "Task": task,
                "Accuracy": f"{metrics['accuracy']:.4f}",
                "F1_Score": f"{metrics['f1']:.4f}",
                "MAE": f"{detailed['mae']:.4f}",
                "Total_Samples": sum(detailed['total_per_class'].values())
            })
    
    # Convert to DataFrame for better visualization
    df = pd.DataFrame(comparison_data)
    print("\n📈 SUMMARY TABLE:")
    print(df.to_string(index=False))
    
    # Find best performing random state
    if len(results_list) > 1:
        print(f"\n🏆 PERFORMANCE ANALYSIS:")
        for task in df['Task'].unique():
            task_df = df[df['Task'] == task]
            best_acc_idx = task_df['Accuracy'].astype(float).idxmax()
            best_f1_idx = task_df['F1_Score'].astype(float).idxmax()
            lowest_mae_idx = task_df['MAE'].astype(float).idxmin()
            
            print(f"\n   {task.upper()} Task:")
            print(f"   • Best Accuracy: rd_state={task_df.loc[best_acc_idx, 'Random_State']} ({task_df.loc[best_acc_idx, 'Accuracy']})")
            print(f"   • Best F1-Score: rd_state={task_df.loc[best_f1_idx, 'Random_State']} ({task_df.loc[best_f1_idx, 'F1_Score']})")
            print(f"   • Lowest MAE: rd_state={task_df.loc[lowest_mae_idx, 'Random_State']} ({task_df.loc[lowest_mae_idx, 'MAE']})")

def main():
    """Main function to run experiments."""
    print("🔬 MULTI-RANDOM-STATE EXPERIMENT RUNNER")
    print("=" * 60)
    
    # Define random states to test
    random_states = [1, 42, 100, 123, 999]
    
    print(f"Will test random states: {random_states}")
    print("Note: Each experiment includes training and testing, which may take significant time.")
    
    # Ask for confirmation
    response = input("\nProceed with experiments? (y/N): ").strip().lower()
    if response != 'y':
        print("Experiments cancelled.")
        return
    
    # Run experiments
    all_results = []
    start_time = datetime.now()
    
    for rd_state in random_states:
        result = run_experiment(rd_state)
        if result:
            all_results.append(result)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n🎉 ALL EXPERIMENTS COMPLETED!")
    print(f"⏱️ Total time: {duration}")
    print(f"✅ Successful experiments: {len(all_results)}/{len(random_states)}")
    
    # Compare results
    if all_results:
        compare_results(all_results)
        
        # Save consolidated results
        consolidated_file = f"results/consolidated_results_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(consolidated_file, 'w') as f:
            json.dump({
                "experiment_info": {
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "duration_seconds": duration.total_seconds(),
                    "tested_random_states": random_states,
                    "successful_experiments": len(all_results)
                },
                "results": all_results
            }, f, indent=2)
        
        print(f"\n💾 Consolidated results saved to: {consolidated_file}")
    
    print(f"\n{'='*60}")
    print("🔍 TO RUN INDIVIDUAL EXPERIMENTS:")
    print("Training: python train_tiktok.py --rd_state <number>")
    print("Testing:  python test_tiktok.py --rd_state <number> --save_predictions")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

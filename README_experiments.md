# Modified TikTok Training and Testing Scripts

This directory contains modified versions of `train_tiktok.py` and `test_tiktok.py` that allow you to:

1. **Configure random state from command line** - to test different train/test splits
2. **Save detailed prediction results** - including MAE, per-class accuracy, confusion matrices, etc.

## Usage

### 1. Training with Custom Random State

```bash
# Train with default random state (123)
python train_tiktok.py

# Train with custom random state
python train_tiktok.py --rd_state 42
python train_tiktok.py --rd_state 100
```

The model checkpoint will be saved with the random state in the filename (e.g., `gpt2_vidmae_whisper_sentiment_rd42_0.pth`).

### 2. Testing with Detailed Results

```bash
# Basic testing (same output as before)
python test_tiktok.py --rd_state 42

# Testing with detailed result saving
python test_tiktok.py --rd_state 42 --save_predictions
```

When using `--save_predictions`, detailed results are saved to `results/test_results_rd<number>.json`.

## Detailed Results Include:

- **Basic metrics**: Accuracy, F1-score
- **MAE (Mean Absolute Error)**: Useful for ordinal classification
- **Per-class accuracy**: How well each sentiment class is predicted
- **Confusion matrix**: Full confusion matrix showing prediction patterns
- **Correct/Total counts**: Number of correct predictions per class
- **Raw predictions and labels**: For further analysis

## Example Output

When running with `--save_predictions`, you'll see:

```
🎯 SENTIMENT TASK RESULTS:
   • Accuracy: 0.7143
   • F1-Score: 0.6895
   • MAE: 0.4286
   • Per-class results:
     - Class 0: 12/15 correct (80.00%)
     - Class 1: 8/12 correct (66.67%)
     - Class 2: 5/8 correct (62.50%)
     - Class 3: 3/7 correct (42.86%)
     - Class 4: 2/3 correct (66.67%)

   • Confusion Matrix:
     True\Pred     0     1     2     3     4
     Class 0:     12     2     1     0     0
     Class 1:      1     8     2     1     0
     Class 2:      0     1     5     2     0
     Class 3:      1     1     1     3     1
     Class 4:      0     0     1     0     2
```

## Experiment Runner

Use `run_experiments.py` to automatically test multiple random states:

```bash
python run_experiments.py
```

This will:
1. Train and test models with different random states
2. Compare results across all experiments
3. Save consolidated results
4. Show which random state performs best

## File Structure

After running experiments, you'll have:

```
results/
├── test_results_rd1.json          # Detailed results for rd_state=1
├── test_results_rd42.json         # Detailed results for rd_state=42
├── test_results_rd100.json        # Detailed results for rd_state=100
└── consolidated_results_<timestamp>.json  # All results combined

checkpoints/
├── gpt2_vidmae_whisper_sentiment_rd1_0.pth    # Model for rd_state=1
├── gpt2_vidmae_whisper_sentiment_rd42_0.pth   # Model for rd_state=42
└── gpt2_vidmae_whisper_sentiment_rd100_0.pth  # Model for rd_state=100
```

## Key Changes Made

### train_tiktok.py
- Added `--rd_state` command line argument
- Added random state to model filename for unique identification
- Prints the random state being used

### test_tiktok.py  
- Added `--rd_state` command line argument
- Added `--save_predictions` flag for detailed result saving
- Calculates additional metrics (MAE, confusion matrix, per-class accuracy)
- Saves comprehensive results to JSON file
- Displays formatted results summary

### iteration.py
- No changes needed (already returns labels and predictions)

## Tips

1. **Different random states** create different train/test splits, helping you understand model stability
2. **MAE is useful** for ordinal sentiment classification (closer predictions are better than distant ones)
3. **Per-class accuracy** helps identify which sentiment classes are hardest to predict
4. **Confusion matrix** shows common misclassification patterns

## Example Workflow

```bash
# Experiment with different data splits
python train_tiktok.py --rd_state 1
python test_tiktok.py --rd_state 1 --save_predictions

python train_tiktok.py --rd_state 100  
python test_tiktok.py --rd_state 100 --save_predictions

# Compare results
python -c "
import json
with open('results/test_results_rd1.json') as f: r1 = json.load(f)
with open('results/test_results_rd100.json') as f: r100 = json.load(f)
print('RD=1 Accuracy:', r1['test_metrics']['sentiment']['accuracy'])
print('RD=100 Accuracy:', r100['test_metrics']['sentiment']['accuracy'])
"
```

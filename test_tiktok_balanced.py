
import torch
import torch.nn as nn
from tokenizers import AddedToken
from transformers import CLIPModel, VideoMAEModel, Wav2Vec2Model, VideoMAEConfig, CLIPConfig, Wav2Vec2Config, XLMRobertaConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSeq2SeqLM
from model.additional_modules import LSTM_fc, FC_head, Gate_Attention
from argparse import Namespace 
from model.model import Multimodal_LLM
from data.dataset import CustomDataset
from iteration import train_model, train_one_epoch, validate
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from transformers import BertTokenizer, AlbertTokenizer, XLMRobertaTokenizerFast, PreTrainedTokenizerFast
from transformers import GPT2Model, BertModel, AlbertModel, XLMRobertaModel
import pickle
from tqdm import tqdm
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
import os
import argparse
from ast import literal_eval
import json

parser = argparse.ArgumentParser(description='Test TikTok sentiment analysis model with balanced metrics')
parser.add_argument('--rd_state', type=int, default=123, help='Random state for train/test split (default: 123)')
parser.add_argument('--save_predictions', action='store_true', help='Save detailed predictions')
parser.add_argument('--model_path', type=str, default=None, help='Path to saved model')
args = parser.parse_args()

tasks_bool = {"engagement" : False, "offensive_level": False, "sentiment" : True}
tasks = []
name = "gpt2_vidmae_whisper_balanced_"

rd_state = args.rd_state
print(f"Testing model with random state: {rd_state}")

for k, v in tasks_bool.items():
    if tasks_bool[k]:
        tasks.append(k)
        name += k + "_"

name += f"rd{rd_state}_"

config = Namespace(
    file_name=name + "0",
    device=torch.device("cuda:0"),
    tokenizer_path="ckpts",
    tasks = tasks,
    engagement_bool = tasks_bool["engagement"],
    offensive_level_bool = tasks_bool["offensive_level"],
    sentiment_bool = tasks_bool["sentiment"],
    video_encoder="MCG-NJU/videomae-base",
    audio_encoder="openai/whisper-small",
    lstm_or_conv = False,
    image_conv_kernel=23,
    image_conv_stride=3,
    image_conv_padding=8,
    video_conv_kernel=36,
    video_conv_stride=24,
    video_conv_padding=0,
    audio_conv_kernel=50,
    audio_conv_stride=23,
    audio_conv_padding=1,
    llm_embed_dim=768,
    llm_output_dim=768,
    attn_dropout=0.1,
    is_add_bias_kv=True,
    is_add_zero_attn=True,
    attention_heads=8,
    image_dim=768,
    video_dim=768,
    audio_dim=768,
    image_seq_len=197,
    video_seq_len=1568,
    audio_seq_len=1500,
    min_mm_seq_len=64,
    lstm_num_layers=1,
    tokenizer_max_len=128,
    add_pooling = False,
    train=False,
    directory = "checkpoints/",
    results_directory = "results/"
)

# Load dataset with same split as training
df = pd.read_csv("video_rating.csv")

# Extract sentiment class labels for stratified splitting
sentiment_labels = []
for idx, row in df.iterrows():
    sentiment_one_hot = literal_eval(row['sentiment'])
    class_idx = sentiment_one_hot.index(1.0) if 1.0 in sentiment_one_hot else 2
    sentiment_labels.append(class_idx)

print(f"\nDataset class distribution:")
unique, counts = np.unique(sentiment_labels, return_counts=True)
for class_idx, count in zip(unique, counts):
    percentage = (count / len(sentiment_labels)) * 100
    print(f"   Class {class_idx}: {count} samples ({percentage:.1f}%)")

# Use stratified split to match training
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=2/3, random_state=rd_state)
train_val_idx, test_idx = next(sss1.split(df, sentiment_labels))

df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
df_test = df.iloc[test_idx].reset_index(drop=True)

train_val_labels = [sentiment_labels[i] for i in train_val_idx]

sss2 = StratifiedShuffleSplit(n_splits=1, test_size=2/3, random_state=rd_state)
train_idx, val_idx = next(sss2.split(df_train_val, train_val_labels))

df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
df_val = df_train_val.iloc[val_idx].reset_index(drop=True)

print(f"\n📊 Test set size: {len(df_test)} samples")

# Check test set class distribution
test_labels = []
for idx, row in df_test.iterrows():
    sentiment_one_hot = literal_eval(row['sentiment'])
    class_idx = sentiment_one_hot.index(1.0) if 1.0 in sentiment_one_hot else 2
    test_labels.append(class_idx)

unique, counts = np.unique(test_labels, return_counts=True)
print(f"\nTest set class distribution:")
for class_idx, count in zip(unique, counts):
    percentage = (count / len(test_labels)) * 100
    print(f"   Class {class_idx}: {count} samples ({percentage:.1f}%)")

batch_size = 4

tokenizer = XLMRobertaTokenizerFast.from_pretrained("roberta-base")
model = XLMRobertaModel.from_pretrained("roberta-base", torch_dtype=torch.float32)
model = Multimodal_LLM(batch_size=batch_size, config=config, tokenizer=tokenizer, adapter_llm=model)

# Load the trained model
model_checkpoint = config.directory + config.file_name + ".pth"
if args.model_path:
    model_checkpoint = args.model_path

if os.path.exists(model_checkpoint):
    print(f"\n📁 Loading model from: {model_checkpoint}")
    model.load_state_dict(torch.load(model_checkpoint, map_location=config.device))
else:
    print(f"\n❌ Model checkpoint not found: {model_checkpoint}")
    exit(1)

test_ds = CustomDataset(df_test, train=False, tokenizer=tokenizer)
test_dataloader = DataLoader(test_ds, batch_size=batch_size, num_workers=8)

# Evaluation with detailed metrics
model = model.to(config.device)
model.eval()

all_predictions = []
all_true_labels = []
all_video_names = []

print(f"\n🧪 Testing model...")

with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(test_dataloader)):
        inputs = {key: value.to(config.device) for key, value in batch.items()}
        outputs = model(inputs)
        
        sentiment_logits = outputs["sentiment"]
        sentiment_predictions = torch.argmax(sentiment_logits, dim=1)
        sentiment_true = torch.argmax(batch["sentiment"], dim=1)
        
        all_predictions.extend(sentiment_predictions.cpu().numpy())
        all_true_labels.extend(sentiment_true.cpu().numpy())
        
        # Get video names for analysis
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(df_test))
        batch_video_names = df_test["v_name"].iloc[start_idx:end_idx].tolist()
        all_video_names.extend(batch_video_names)

# Calculate comprehensive metrics
print(f"\n📊 DETAILED EVALUATION RESULTS:")
print("="*60)

# Standard accuracy
accuracy = np.mean(np.array(all_predictions) == np.array(all_true_labels))
print(f"Standard Accuracy: {accuracy:.4f}")

# Balanced accuracy (accounts for class imbalance)
balanced_acc = balanced_accuracy_score(all_true_labels, all_predictions)
print(f"Balanced Accuracy: {balanced_acc:.4f}")

# Per-class metrics
class_report = classification_report(all_true_labels, all_predictions, 
                                   target_names=[f"Class {i}" for i in range(5)],
                                   output_dict=True)

print(f"\n📋 CLASSIFICATION REPORT:")
print(classification_report(all_true_labels, all_predictions, 
                          target_names=[f"Class {i}" for i in range(5)]))

# Confusion matrix
cm = confusion_matrix(all_true_labels, all_predictions)
print(f"\n🔢 CONFUSION MATRIX:")
print("Predicted →")
print("     ", end="")
for i in range(5):
    print(f"  C{i}", end="")
print()

for i in range(5):
    print(f"C{i} |", end="")
    for j in range(5):
        print(f"{cm[i,j]:4d}", end="")
    print(f" | {np.sum(cm[i,:])}")

print("     ", end="")
for i in range(5):
    print("----", end="")
print()
print("     ", end="")
for i in range(5):
    print(f"{np.sum(cm[:,i]):4d}", end="")
print(f" | {np.sum(cm)}")

# Check if model is still predicting only one class
unique_preds, pred_counts = np.unique(all_predictions, return_counts=True)
print(f"\n🎯 PREDICTION DISTRIBUTION:")
for pred_class, count in zip(unique_preds, pred_counts):
    percentage = (count / len(all_predictions)) * 100
    print(f"   Predicted Class {pred_class}: {count} times ({percentage:.1f}%)")

if len(unique_preds) == 1:
    print(f"\n❌ WARNING: Model still predicts only Class {unique_preds[0]}!")
    print("   Consider:")
    print("   • Increasing learning rate")
    print("   • Using different class weights")
    print("   • Adding data augmentation")
    print("   • Using focal loss instead of weighted CE")
else:
    print(f"\n✅ SUCCESS: Model predicts {len(unique_preds)} different classes!")

# Save detailed results
results = {
    "random_state": rd_state,
    "test_size": len(df_test),
    "accuracy": accuracy,
    "balanced_accuracy": balanced_acc,
    "classification_report": class_report,
    "confusion_matrix": cm.tolist(),
    "prediction_distribution": {int(k): int(v) for k, v in zip(unique_preds, pred_counts)}
}

if args.save_predictions:
    detailed_results = []
    for i, (true_label, pred_label, video_name) in enumerate(zip(all_true_labels, all_predictions, all_video_names)):
        detailed_results.append({
            "sample_id": i,
            "video_name": video_name,
            "true_label": int(true_label),
            "predicted_label": int(pred_label),
            "correct": int(true_label) == int(pred_label)
        })
    
    results["detailed_predictions"] = detailed_results

# Save results
output_file = f"results/test_results_balanced_rd{rd_state}.json"
os.makedirs("results", exist_ok=True)

with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Results saved to: {output_file}")

# Final recommendations
print(f"\n💡 RECOMMENDATIONS:")
if balanced_acc < 0.3:
    print("   ❌ Very poor performance. Try:")
    print("      • Focal loss with higher gamma (2-5)")
    print("      • More aggressive data augmentation")
    print("      • Lower learning rate with more epochs")
elif balanced_acc < 0.5:
    print("   ⚠️  Poor performance. Try:")
    print("      • Fine-tune class weights")
    print("      • Increase model capacity")
    print("      • Add regularization")
elif balanced_acc < 0.7:
    print("   🟡 Moderate performance. Try:")
    print("      • Fine-tune hyperparameters")
    print("      • Add ensemble methods")
else:
    print("   ✅ Good performance!")

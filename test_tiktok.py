'''
python -W ignore run.py
python block_core.py
find -type d -name 'pymp*' -exec rm -r {} \;
'''

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
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, AlbertTokenizer, XLMRobertaTokenizerFast, PreTrainedTokenizerFast #only for gpt2 and assign values
from transformers import GPT2Model, BertModel, AlbertModel, XLMRobertaModel
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import os
import argparse
import json
from sklearn.metrics import mean_absolute_error, confusion_matrix
import numpy as np

# Add command line argument parsing
parser = argparse.ArgumentParser(description='Test TikTok sentiment analysis model')
parser.add_argument('--rd_state', type=int, default=123, help='Random state for train/test split (default: 123)')
parser.add_argument('--save_predictions', action='store_true', help='Save detailed prediction results')
args = parser.parse_args()

tasks_bool = {"engagement" : True, "offensive_level": False, "sentiment" : False}
tasks = []
name = "gpt2_vidmae_whisper_"

rd_state = args.rd_state
print(f"Using random state: {rd_state}")

for k, v in tasks_bool.items():
    if tasks_bool[k]:
        tasks.append(k)
        name += k + "_"

# Add random state to filename to load the correct model
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
    train=True,
    directory = "checkpoints/",
    results_directory = "results/"
)

df = pd.read_csv("video_rating.csv")


df_train_val, df_test = train_test_split(df, test_size=2/3, random_state=rd_state)
df_train, df_val = train_test_split(df_train_val, test_size=2/3, random_state=rd_state)

num_epochs = 3
patience = 10
batch_size = 4

#for roberta
tokenizer = XLMRobertaTokenizerFast.from_pretrained("roberta-base")
model = XLMRobertaModel.from_pretrained("roberta-base", torch_dtype=torch.float32)

#for gpt2

# tokenizer = PreTrainedTokenizerFast.from_pretrained('l3cube-pune/hing-gpt')
# model = GPT2Model.from_pretrained('l3cube-pune/hing-gpt', torch_dtype=torch.float32)
# tokenizer.bos_token_id = 1
# tokenizer.eos_token_id = 2


model = Multimodal_LLM(batch_size=batch_size, config=config, tokenizer=tokenizer, adapter_llm=model)

train_ds = CustomDataset(dataframe=df_train, train=True, tokenizer=tokenizer)
val_ds = CustomDataset(df_val, train=True, tokenizer=tokenizer)
test_ds = CustomDataset(df_test, train=False, tokenizer=tokenizer)

train_dataloader = DataLoader(train_ds, batch_size=batch_size, num_workers=8, shuffle=True)
val_dataloader = DataLoader(val_ds, batch_size=batch_size, num_workers=8)
test_dataloader = DataLoader(test_ds, batch_size=batch_size, num_workers=8)

checkpoint_path = config.directory + config.file_name + ".pth"

state_dict = torch.load(checkpoint_path, map_location=config.device)
print("load state dict",checkpoint_path)
model.load_state_dict(state_dict)

# Get detailed test results
test_metrics, all_labels, all_predictions = validate(model, test_dataloader, config)

# Calculate additional metrics and save detailed results if requested
if args.save_predictions:
    detailed_results = {
        "random_state": rd_state,
        "model_file": config.file_name,
        "test_metrics": test_metrics,
        "detailed_analysis": {}
    }
    
    for task in config.tasks:
        labels = np.array(all_labels[task])
        predictions = np.array(all_predictions[task])
        
        # Calculate MAE
        mae = mean_absolute_error(labels, predictions)
        
        # Calculate confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Calculate per-class accuracy
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        
        # Count correct predictions per class
        unique_labels = np.unique(labels)
        correct_per_class = {}
        total_per_class = {}
        
        for label in unique_labels:
            mask = labels == label
            correct_per_class[int(label)] = int(np.sum(predictions[mask] == label))
            total_per_class[int(label)] = int(np.sum(mask))
        
        detailed_results["detailed_analysis"][task] = {
            "mae": float(mae),
            "confusion_matrix": cm.tolist(),
            "per_class_accuracy": per_class_accuracy.tolist(),
            "correct_per_class": correct_per_class,
            "total_per_class": total_per_class,
            "predictions": predictions.tolist(),
            "true_labels": labels.tolist()
        }
    
    # Save results
    results_filename = f"results/test_results_rd{rd_state}.json"
    with open(results_filename, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\n📊 DETAILED RESULTS SAVED TO: {results_filename}")
    print("="*60)
    
    for task in config.tasks:
        print(f"\n🎯 {task.upper()} TASK RESULTS:")
        print(f"   • Accuracy: {test_metrics[task]['accuracy']:.4f}")
        print(f"   • F1-Score: {test_metrics[task]['f1']:.4f}")
        print(f"   • MAE: {detailed_results['detailed_analysis'][task]['mae']:.4f}")
        
        print(f"   • Per-class results:")
        for label, correct in detailed_results['detailed_analysis'][task]['correct_per_class'].items():
            total = detailed_results['detailed_analysis'][task]['total_per_class'][label]
            acc = correct / total if total > 0 else 0
            print(f"     - Class {label}: {correct}/{total} correct ({acc:.2%})")
        
        print(f"\n   • Confusion Matrix:")
        cm = np.array(detailed_results['detailed_analysis'][task]['confusion_matrix'])
        print("     True\\Pred", end="")
        for i in range(cm.shape[1]):
            print(f"{i:6}", end="")
        print()
        for i, row in enumerate(cm):
            print(f"     Class {i}:", end="")
            for val in row:
                print(f"{val:6}", end="")
            print()

else:
    print("\nRun with --save_predictions to save detailed results")
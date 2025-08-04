
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
from transformers import BertTokenizer, AlbertTokenizer, XLMRobertaTokenizerFast, PreTrainedTokenizerFast
from transformers import GPT2Model, BertModel, AlbertModel, XLMRobertaModel
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import os
import argparse
from ast import literal_eval

# Add command line argument parsing
parser = argparse.ArgumentParser(description='Train TikTok sentiment analysis model with balanced classes')
parser.add_argument('--rd_state', type=int, default=123, help='Random state for train/test split (default: 123)')
parser.add_argument('--use_weighted_loss', action='store_true', default=True, help='Use weighted cross-entropy loss')
parser.add_argument('--use_stratified_split', action='store_true', default=True, help='Use stratified sampling for train/val/test split')
args = parser.parse_args()

tasks_bool = {"engagement" : False, "offensive_level": False, "sentiment" : True}
tasks = []
name = "gpt2_vidmae_whisper_balanced_"

rd_state = args.rd_state
print(f"Using random state: {rd_state}")
print(f"Using weighted loss: {args.use_weighted_loss}")
print(f"Using stratified split: {args.use_stratified_split}")

for k, v in tasks_bool.items():
    if tasks_bool[k]:
        tasks.append(k)
        name += k + "_"

# Add random state to filename to distinguish different splits
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

# Load dataset
df = pd.read_csv("video_rating.csv")

# Extract sentiment class labels for stratified splitting
sentiment_labels = []
for idx, row in df.iterrows():
    sentiment_one_hot = literal_eval(row['sentiment'])
    class_idx = sentiment_one_hot.index(1.0) if 1.0 in sentiment_one_hot else 2  # Default to class 2 if invalid
    sentiment_labels.append(class_idx)

print(f"\nDataset class distribution:")
unique, counts = np.unique(sentiment_labels, return_counts=True)
for class_idx, count in zip(unique, counts):
    percentage = (count / len(sentiment_labels)) * 100
    print(f"   Class {class_idx}: {count} samples ({percentage:.1f}%)")

if args.use_stratified_split:
    print("\n🔄 Using stratified sampling for balanced train/val/test splits...")
    
    # First split: train+val vs test (stratified)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=2/3, random_state=rd_state)
    train_val_idx, test_idx = next(sss1.split(df, sentiment_labels))
    
    df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)
    
    # Extract labels for the train_val subset
    train_val_labels = [sentiment_labels[i] for i in train_val_idx]
    
    # Second split: train vs val (stratified)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=2/3, random_state=rd_state)
    train_idx, val_idx = next(sss2.split(df_train_val, train_val_labels))
    
    df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
    df_val = df_train_val.iloc[val_idx].reset_index(drop=True)
    
else:
    print("\n🔄 Using random sampling for train/val/test splits...")
    df_train_val, df_test = train_test_split(df, test_size=(2/3), random_state=rd_state)
    df_train, df_val = train_test_split(df_train_val, test_size=(2/3), random_state=rd_state)

# Print split statistics
print(f"\n📊 Split sizes:")
print(f"   Train: {len(df_train)} samples")
print(f"   Validation: {len(df_val)} samples") 
print(f"   Test: {len(df_test)} samples")

# Check class distribution in each split
for split_name, split_df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
    split_labels = []
    for idx, row in split_df.iterrows():
        sentiment_one_hot = literal_eval(row['sentiment'])
        class_idx = sentiment_one_hot.index(1.0) if 1.0 in sentiment_one_hot else 2
        split_labels.append(class_idx)
    
    unique, counts = np.unique(split_labels, return_counts=True)
    print(f"\n{split_name} class distribution:")
    for class_idx, count in zip(unique, counts):
        percentage = (count / len(split_labels)) * 100
        print(f"   Class {class_idx}: {count} samples ({percentage:.1f}%)")

num_epochs = 3  # Increase epochs for better convergence
patience = 10
batch_size = 4

# For roberta
tokenizer = XLMRobertaTokenizerFast.from_pretrained("roberta-base")
model = XLMRobertaModel.from_pretrained("roberta-base", torch_dtype=torch.float32)

model = Multimodal_LLM(batch_size=batch_size, config=config, tokenizer=tokenizer, adapter_llm=model)

train_ds = CustomDataset(dataframe=df_train, train=True, tokenizer=tokenizer)
val_ds = CustomDataset(df_val, train=True, tokenizer=tokenizer)
test_ds = CustomDataset(df_test, train=False, tokenizer=tokenizer)

train_dataloader = DataLoader(train_ds, batch_size=batch_size, num_workers=8, shuffle=True)
val_dataloader = DataLoader(val_ds, batch_size=batch_size, num_workers=8)
test_dataloader = DataLoader(test_ds, batch_size=batch_size, num_workers=8)

print(f"\n🚀 Starting training with balanced approach...")
train_model(model, train_dataloader, val_dataloader, config, num_epochs, "sentiment", "f1", devices=None)

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

# Add command line argument parsing
parser = argparse.ArgumentParser(description='Train TikTok sentiment analysis model')
parser.add_argument('--rd_state', type=int, default=123, help='Random state for train/test split (default: 123)')
args = parser.parse_args()

tasks_bool = {"offensive" : False, "offensive_level": False, "sentiment" : True}
tasks = []
name = "gpt2_vidmae_whisper_"

rd_state = args.rd_state
print(f"Using random state: {rd_state}")

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
    offensive_bool = tasks_bool["offensive"],
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


df = pd.read_csv("tiktok_data/video_rating.csv")
#df=df.head(100)
df_train_val, df_test = train_test_split(df, test_size=(2/3), random_state=rd_state)
df_train, df_val = train_test_split(df_train_val, test_size=(2/3), random_state=rd_state)

num_epochs = 3
patience = 10
batch_size = 4

#for roberta
tokenizer = XLMRobertaTokenizerFast.from_pretrained("l3cube-pune/hing-roberta")
model = XLMRobertaModel.from_pretrained("l3cube-pune/hing-roberta", torch_dtype=torch.float32)

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


train_model(model, train_dataloader, val_dataloader, config, num_epochs, "sentiment", "f1", devices=None)

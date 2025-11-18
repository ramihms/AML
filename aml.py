import os
import warnings
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

class Config:
    BASE_PATH = "/kaggle/input/aml-competition"
    TRAIN_NPZ = f"{BASE_PATH}/train/train/train.npz"
    TEST_NPZ = f"{BASE_PATH}/test/test/test.clean.npz"
    OUTPUT_BASE = "/kaggle/working"
    MODEL_SAVE_PATH = f"{OUTPUT_BASE}/sota_translator_best.pth"
    SUBMISSION_FILE = f"{OUTPUT_BASE}/sota_submission.csv"
    TEXT_DIM = 1024
    IMAGE_DIM = 1536
    HIDDEN_DIM = 2048
    NUM_BLOCKS = 8
    NUM_HEADS = 8
    BATCH_SIZE = 128
    NUM_EPOCHS = 120
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    NOISE_STD = 0.015
    PRINT_EVERY = 100
    SAVE_EVERY = 10

os.makedirs(Config.OUTPUT_BASE, exist_ok=True)
os.makedirs(f"{Config.OUTPUT_BASE}/checkpoints", exist_ok=True)

class NPZDataset(Dataset):
    """Dataset embeddings + augmentation"""
    def __init__(self, npz_file, augment=True, noise_std=0.01):
        data = np.load(npz_file)
        self.text_embeddings = torch.tensor(data['captions/embeddings'], dtype=torch.float32)
        labels = data['captions/label']    # (n, 25000), one-hot
        image_embeddings = torch.tensor(data['images/embeddings'], dtype=torch.float32)
        self.image_embeddings = torch.stack([
            image_embeddings[np.argmax(labels[i])] for i in range(len(labels))
        ])
        self.augment = augment
        self.noise_std = noise_std
    def __len__(self):
        return len(self.text_embeddings)
    def __getitem__(self, idx):
        text = self.text_embeddings[idx].clone()
        image = self.image_embeddings[idx].clone()
        if self.augment and torch.rand(1).item() > 0.5:
            noise = torch.randn_like(text) * self.noise_std
            text = text + noise
        return text, image

class AttentionResidualBlock(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.12):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(dim)
    def forward(self, x):
        attn_out, _ = self.attn(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
        x = x + attn_out.squeeze(1)
        x = self.ln1(x)
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        return x

class SOTA_Translator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_blocks, n_heads):
        super().__init__()
        self.fc_in = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.blocks = nn.Sequential(*[
            AttentionResidualBlock(hidden_dim, heads=n_heads, dropout=0.12) for _ in range(n_blocks)
        ])
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
    def forward(self, x):
        x = self.fc_in(x)
        x = self.blocks(x)
        x = self.fc_out(x)
        x = x / (x.norm(dim=1, keepdim=True) + 1e-8)  # L2-normalisation
        return x

def info_nce_loss(pred, target, temperature=0.07):
    pred = nn.functional.normalize(pred, dim=1)
    target = nn.functional.normalize(target, dim=1)
    logits = torch.mm(pred, target.t()) / temperature
    labels = torch.arange(len(pred)).to(pred.device)
    return nn.CrossEntropyLoss()(logits, labels)

def train_epoch(model, loader, optimizer, device, epoch, num_epochs):
    model.train()
    total_loss = 0
    num_batches = len(loader)
    start_time = time.time()
    for i, (text, image) in enumerate(loader):
        text = text.to(device)
        image = image.to(device)
        optimizer.zero_grad()
        out = model(text)
        loss = info_nce_loss(out, image, temperature=0.07)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        if (i+1) % Config.PRINT_EVERY == 0:
            avg_loss = total_loss / (i+1)
            eta_sec = ((num_batches - i - 1) * (time.time()-start_time) / (i+1))
            print(f"  [{i+1:4d}/{num_batches}] Loss: {loss.item():.4f} (avg: {avg_loss:.4f}) | ETA: {eta_sec/60:.2f}min")
    avg_loss = total_loss / num_batches
    print(f"✓ Epoch {epoch+1} complete: Avg Loss {avg_loss:.4f}")
    return avg_loss

def train_model(model, loader, num_epochs, device, lr):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr * 2,
        epochs=num_epochs,
        steps_per_epoch=len(loader),
        pct_start=0.1,
        anneal_strategy='cos'
    )
    best_loss = float('inf')
    losses = []
    for epoch in range(num_epochs):
        avg_loss = train_epoch(model, loader, optimizer, device, epoch, num_epochs)
        losses.append(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, Config.MODEL_SAVE_PATH)
            print(f"  💾 Best model saved (epoch {epoch+1}, loss {avg_loss:.4f})")
        if (epoch+1)%Config.SAVE_EVERY==0:
            torch.save(model.state_dict(), f"{Config.OUTPUT_BASE}/checkpoints/epoch_{epoch+1}.pth")
            print(f"  Checkpoint: epoch_{epoch+1}.pth")
        torch.cuda.empty_cache()
    plt.figure(figsize=(10,6)); plt.plot(losses)
    plt.title("Loss curve (InfoNCE contrastive loss)")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(True)
    plt.savefig(f"{Config.OUTPUT_BASE}/sota_training_loss.png", dpi=150)
    plt.close()
    print(f"Finished training! Best val loss: {best_loss:.4f}")
    return model

def generate_submission(model, test_npz, device, output_file):
    test_data = np.load(test_npz)
    test_ids = test_data['captions/ids']
    test_text = torch.tensor(test_data['captions/embeddings'], dtype=torch.float32).to(device)
    model.eval()
    preds = []
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(test_text), batch_size):
            batch = test_text[i:i+batch_size]
            out = model(batch)
            preds.append(out.cpu().numpy())
    all_preds = np.vstack(preds)
    results = []
    for idx, emb in zip(test_ids, all_preds):
        results.append({
            'id': int(idx),
            'embedding': json.dumps(emb.tolist())
        })
    sub_df = pd.DataFrame(results)
    sub_df.to_csv(output_file, index=False)
    print(f"Submission saved: {output_file} ({len(sub_df)} rows)")
    print(sub_df.head())
    return sub_df

print("Loading NPZ data...")
dataset = NPZDataset(Config.TRAIN_NPZ, augment=True, noise_std=Config.NOISE_STD)
loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

print("Building SOTA Translator...")
model = SOTA_Translator(
    input_dim=Config.TEXT_DIM, hidden_dim=Config.HIDDEN_DIM,
    output_dim=Config.IMAGE_DIM, n_blocks=Config.NUM_BLOCKS, n_heads=Config.NUM_HEADS
)
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

print("Training...")
best_model = train_model(model, loader, Config.NUM_EPOCHS, device, Config.LEARNING_RATE)

print("Generating Submission...")
submission_df = generate_submission(
    best_model, Config.TEST_NPZ, device, Config.SUBMISSION_FILE
)

print("Done! Ready to submit.")

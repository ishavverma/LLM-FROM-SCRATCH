import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from src.model.tokenizer import BPETokenizer
from src.model.transformer import GPTLanguageModel
from src.model.lora import inject_lora, mark_only_lora_as_trainable
import math
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class TextDataset(Dataset):
    """Dataset yielding (input, target) pairs for autoregressive training, with instruct masking."""
    def __init__(self, data_path, tokenizer, block_size):
        self.block_size = block_size
        self.tokenizer = tokenizer
        
        # Load dataset
        with open(data_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
            
        # Train tokenizer if it hasn't been trained yet
        text_for_training = ""
        for c in chunks:
            if isinstance(c, str): text_for_training += c + "\n"
            else: text_for_training += c.get("text", "") + "\n"
            
        if not tokenizer.vocab or len(tokenizer.vocab) <= 256:
            tokenizer.train(text_for_training)
            
        self.all_token_ids = []
        self.all_loss_targets = []
        
        for c in chunks:
            chunk_type = "base"
            chunk_text = ""
            if isinstance(c, str):
                chunk_text = c
            else:
                chunk_text = c.get("text", "")
                chunk_type = c.get("type", "base")
                
            tokens = tokenizer.encode(chunk_text)
            
            if chunk_type in ["instruct", "chat"]:
                targets = []
                mask_on = False
                for t in tokens:
                    if t == tokenizer.special_tokens.get("<|response|>") or t == tokenizer.special_tokens.get("<|assistant|>"):
                        mask_on = True
                        targets.append(-100) # ignore loss on the tag itself
                    elif t == tokenizer.special_tokens.get("<|endoftext|>"):
                        targets.append(t)
                        mask_on = False # end masking
                    else:
                        targets.append(t if mask_on else -100)
            else:
                targets = tokens.copy()
                
            self.all_token_ids.extend(tokens)
            self.all_loss_targets.extend(targets)
            
            eot = tokenizer.special_tokens.get("<|endoftext|>", -100)
            self.all_token_ids.append(eot)
            self.all_loss_targets.append(eot)
        
    def __len__(self):
        return max(0, len(self.all_token_ids) - self.block_size)
        
    def __getitem__(self, idx):
        x = torch.tensor(self.all_token_ids[idx : idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.all_loss_targets[idx + 1 : idx + self.block_size + 1], dtype=torch.long)
        return x, y

def train(data_path="data/approved_chunks.json", epochs=1, batch_size=4, block_size=256, learning_rate=1e-3, device='cpu', apply_lora=False):
    print(f"Training on device: {device}")
    
    # Initialize tokenizer and dataset
    tokenizer = BPETokenizer(vocab_size=1000)
    full_dataset = TextDataset(data_path, tokenizer, block_size)
    
    # Split into train/val
    val_size = max(1, int(len(full_dataset) * 0.1))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model using total vocab size mapping
    vocab_size = tokenizer.total_vocab_size
    model = GPTLanguageModel(vocab_size=vocab_size, 
                             d_model=128, 
                             n_heads=4, 
                             n_kv_heads=2, 
                             n_layer=2, 
                             block_size=block_size)
                             
    if apply_lora:
        print("Injecting LoRA adapters...")
        inject_lora(model, rank=8, alpha=16)
        mark_only_lora_as_trainable(model)
        
    model.to(device)
    
    # Optimizer and AMP features
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader), eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None 
    grad_accum_steps = 4 
    
    best_val_loss = float('inf')
    early_stopping_patience = 3
    epochs_no_improve = 0
    writer = SummaryWriter(log_dir="runs/experiment_1")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        optimizer.zero_grad()
        for step, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            
            # AMP
            if device == "cuda":
                with torch.cuda.amp.autocast(): 
                    logits, loss, _ = model(x, targets=y)
                loss = loss / grad_accum_steps
                scaler.scale(loss).backward()
                
                if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            else:
                logits, loss, _ = model(x, targets=y)
                loss = loss / grad_accum_steps
                loss.backward()
                
                if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            
            accumulated_loss = loss.item() * grad_accum_steps
            total_loss += accumulated_loss
            pbar.set_postfix({'loss': f"{accumulated_loss:.4f}"})
            
            global_step = epoch * len(train_loader) + step
            writer.add_scalar("Loss/train_step", accumulated_loss, global_step)
            
        avg_train_loss = total_loss / len(train_loader)
        train_ppl = math.exp(min(avg_train_loss, 20))
        writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)
        writer.add_scalar("Perplexity/train_epoch", train_ppl, epoch)
        
        # Validation Loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                if device == "cuda":
                    with torch.cuda.amp.autocast(): # Corrected autocast usage
                        _, loss, _ = model(x, targets=y)
                else:
                    _, loss, _ = model(x, targets=y)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        val_ppl = math.exp(min(avg_val_loss, 20))
        writer.add_scalar("Loss/val_epoch", avg_val_loss, epoch)
        writer.add_scalar("Perplexity/val_epoch", val_ppl, epoch)
        
        print(f"Epoch {epoch+1} finished. Train Loss: {avg_train_loss:.4f} (PPL: {train_ppl:.2f}) | Val Loss: {avg_val_loss:.4f} (PPL: {val_ppl:.2f})")
        
        # Save checkpoints & Early Stopping
        os.makedirs("models", exist_ok=True)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            print(f"New best validation loss! Saving model...")
            torch.save(model.state_dict(), "models/transformer_weights.pth")
            tokenizer.save("models/tokenizer.json")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= early_stopping_patience:
                print("Early stopping triggered. Halting training.")
                break
                
    writer.close()
    print("Training complete.")

if __name__ == "__main__":
    train()

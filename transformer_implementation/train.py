import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from utils import create_masks, get_std_opt


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_tokens = 0
    
    print(f"Training on {len(dataloader)} batches...")
    
    for i, batch in enumerate(dataloader):
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        src_mask, tgt_mask = create_masks(src, tgt_input)
        
        optimizer.zero_grad()
        
        output = model(src, tgt_input, src_mask, tgt_mask)
        loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_tokens += (tgt_output != 0).sum().item()
        
        if (i + 1) % 50 == 0:
            avg_loss = total_loss / (i + 1)
            curr_lr = optimizer._rate if hasattr(optimizer, '_rate') else optimizer.param_groups[0]['lr']
            print(f"  Batch {i+1}/{len(dataloader)} | Loss: {avg_loss:.4f} | LR: {curr_lr:.6f}")
        
        if (i + 1) % 200 == 0:
            perplexity = torch.exp(torch.tensor(total_loss / (i + 1)))
            print(f"  Current Perplexity: {perplexity:.2f}")
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    print(f"Evaluating on {len(dataloader)} batches...")
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            src_mask, tgt_mask = create_masks(src, tgt_input)
            
            output = model(src, tgt_input, src_mask, tgt_mask)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            
            total_loss += loss.item()
            
            if (i + 1) % 20 == 0:
                print(f"  Eval Batch {i+1}/{len(dataloader)}")
    
    return total_loss / len(dataloader)


def train_transformer(model, train_loader, val_loader, n_epochs, device, 
                     pad_idx=0, checkpoint_path=None):
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = get_std_opt(model)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print(f"\nStarting training for {n_epochs} epochs...")
    print(f"Total training batches: {len(train_loader)}")
    print(f"Total validation batches: {len(val_loader)}")
    print("-" * 70)
    
    for epoch in range(n_epochs):
        start_time = time.time()
        
        print(f"\nEpoch {epoch+1}/{n_epochs}")
        print("-" * 50)
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        
        train_ppl = torch.exp(torch.tensor(train_loss))
        val_ppl = torch.exp(torch.tensor(val_loss))
        
        print(f"\nEpoch Summary:")
        print(f"  Time: {int(epoch_mins)}m {int(epoch_secs)}s")
        print(f"  Train Loss: {train_loss:.3f} | Train PPL: {train_ppl:.2f}")
        print(f"  Val Loss: {val_loss:.3f} | Val PPL: {val_ppl:.2f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if checkpoint_path:
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  ✓ New best model saved! (Val Loss: {val_loss:.3f})")
        else:
            print(f"  Current best Val Loss: {best_val_loss:.3f}")
        
        if epoch > 5 and val_losses[-1] > val_losses[-2] > val_losses[-3]:
            print("\n⚠️  Validation loss increasing for 3 epochs - early stopping might be needed!")
        
        print("=" * 70)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = Transformer(
        src_vocab_size=10000,
        tgt_vocab_size=10000,
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        max_len=100,
        dropout=0.1
    ).to(device)
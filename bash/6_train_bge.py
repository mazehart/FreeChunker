import random
import numpy as np
import torch
from datasets import load_from_disk
from src.freechunker import FreeChunkerModel
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import json
from transformers import get_cosine_schedule_with_warmup
from utils.monitor import Monitor
from tqdm import tqdm

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Initialize GPU monitor
gpu_monitor = Monitor(device_id="1")
gpu_monitor.setup()

# Load vector_datasets data
train_dataset = load_from_disk('./vector/BAAI--bge-m3/train')
# Load validation set data (assuming same structure as training set)
val_dataset_full = load_from_disk('./vector/BAAI--bge-m3/val')
# Limit validation set to 200 samples
val_dataset = val_dataset_full.select(range(min(200, len(val_dataset_full))))
print(f"Validation set samples: {len(val_dataset)}")

# Dataset objects from datasets can be used directly with DataLoader, but must implement __getitem__ and __len__
class ArrowDataset(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        item = self.ds[idx]
        return {
            'input_ids': torch.tensor(item['input'], dtype=torch.float).to('cuda'),
            'labels': torch.tensor(item['label'], dtype=torch.float).to('cuda')
        }

arrow_dataset = ArrowDataset(train_dataset)
train_loader = DataLoader(arrow_dataset, batch_size=1, shuffle=True)

# Create validation DataLoader
val_arrow_dataset = ArrowDataset(val_dataset)
val_loader = DataLoader(val_arrow_dataset, batch_size=1, shuffle=False)

print('Loading model...')
# 1. Load pretrained model
model = FreeChunkerModel.from_pretrained('/share/home/ecnuzwx/UnifiedRAG/cache/models--BAAI--bge-m3', ignore_mismatched_sizes=True)
model = model.to('cuda')
print('Model loaded')

# 2. Training
# Optimize only unfrozen parameters
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Calculate total steps
num_epochs = 2
num_training_steps = num_epochs * len(train_loader)
num_warmup_steps = num_training_steps // 3

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

train_losses = []
step_losses = []
val_losses = []
val_step_losses = []
eval_interval = 1000

save_dir = f'./saved_models/bge-m3'
os.makedirs(save_dir, exist_ok=True)



def evaluate_validation(model, val_loader):
    """Evaluate validation set"""
    model.eval()
    val_loss_sum = 0
    val_count = 0
    
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc="Validating", unit="batch", leave=False)
        for batch in val_pbar:
            outputs = model(
                inputs_embeds=batch['input_ids'],
                labels=batch['labels']
            )
            loss = outputs['loss'].item()
            val_loss_sum += loss
            val_count += 1
            
            # Update validation progress bar info
            val_pbar.set_postfix({'val_loss': f'{loss:.4f}'})
    
    avg_val_loss = val_loss_sum / val_count if val_count > 0 else 0
    model.train()  # Switch back to training mode
    return avg_val_loss

def plot_training_loss(current_step, epoch, step_losses, train_losses, val_step_losses, val_losses, model, val_loader):
    avg_train_loss = sum(step_losses) / len(step_losses) if step_losses else 0
    train_losses.append(avg_train_loss)
    
    # Evaluate validation set
    avg_val_loss = evaluate_validation(model, val_loader)
    val_losses.append(avg_val_loss)
    val_step_losses.append(avg_val_loss)
    
    print(f"Epoch {epoch}, Step {current_step}, avg train loss: {avg_train_loss:.4f}, avg val loss: {avg_val_loss:.4f}")
    
    # Plotting
    plt.figure(figsize=(12, 8))
    
    # Training loss: plotted per step (continuous curve)
    train_steps = list(range(1, len(step_losses) + 1))
    plt.plot(train_steps, step_losses, label='Train Loss (per step)', color='blue', alpha=0.7, linewidth=0.8)
    
    # Validation loss: plotted per eval interval (continuous line)
    eval_steps = list(range(eval_interval, len(step_losses) + 1, eval_interval))
    if len(eval_steps) == len(val_losses):
        plt.plot(eval_steps, val_losses, label='Validation Loss (eval interval)', color='red', linewidth=2)
    
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss (per step) vs Validation Loss (eval interval)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()

# Training main loop
for epoch in range(num_epochs):
    train_pbar = tqdm(enumerate(train_loader, 1), total=len(train_loader), 
                      desc=f"Epoch {epoch+1}/{num_epochs}", 
                      unit="batch")
    
    for step, batch in train_pbar:
        outputs = model(
            inputs_embeds=batch['input_ids'],
            labels=batch['labels']
        )
        loss = outputs['loss']
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        step_losses.append(loss.item())
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update progress bar info
        train_pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{current_lr:.2e}'
        })

        if step % eval_interval == 0:
            plot_training_loss(step, epoch, step_losses, train_losses, val_step_losses, val_losses, model, val_loader)
    
    # Save model at the end of each epoch
    model_save_path = os.path.join(save_dir, f'xlmroberta_epoch_{epoch}')
    model.save_pretrained(model_save_path)
    print(f"Saved model for epoch {epoch}")

# Plot complete curve one last time
plt.figure(figsize=(12, 8))

# Training loss: plotted per step (continuous curve)
train_steps = list(range(1, len(step_losses) + 1))
plt.plot(train_steps, step_losses, label='Train Loss (per step)', color='blue', alpha=0.7, linewidth=0.8)

# Validation loss: plotted per eval interval (continuous line)
eval_steps = list(range(eval_interval, len(step_losses) + 1, eval_interval))
if len(eval_steps) == len(val_losses):
    plt.plot(eval_steps, val_losses, label='Validation Loss (eval interval)', color='red', linewidth=2)

plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.legend()
plt.title('Final Training Loss (per step) vs Validation Loss (eval interval)')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(save_dir, 'final_loss_curve.png'), dpi=150, bbox_inches='tight')
plt.show() 

# Save training and validation loss data as JSON file
training_data = {
    'training_losses': {
        'step_losses': step_losses,  # Training loss per step
        'steps': list(range(1, len(step_losses) + 1))  # Corresponding steps
    },
    'validation_losses': {
        'val_losses': val_losses,  # Validation loss values
        'eval_steps': list(range(eval_interval, len(step_losses) + 1, eval_interval))  # Corresponding eval steps
    },
    'training_config': {
        'eval_interval': eval_interval,
        'num_epochs': num_epochs,
        'total_steps': len(step_losses)
    },
    'summary': {
        'final_train_loss': step_losses[-1] if step_losses else None,
        'final_val_loss': val_losses[-1] if val_losses else None,
        'min_train_loss': min(step_losses) if step_losses else None,
        'min_val_loss': min(val_losses) if val_losses else None
    }
}

training_data_path = os.path.join(save_dir, 'training_losses.json')
with open(training_data_path, 'w') as f:
    json.dump(training_data, f, indent=2)
print(f"Saved complete training and validation losses to {training_data_path}")
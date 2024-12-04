import numpy as np
import torch
from transformers import BertTokenizer, BertForMaskedLM
from torch.utils.data import DataLoader, TensorDataset
from metrics import *
# Step 1: Load the token IDs
token_ids = np.load('bookcorpus_token_ids.npy')  # Adjust the filename if necessary

# Convert to PyTorch tensor
input_ids = torch.tensor(token_ids)
print(f"Loaded token IDs with shape: {input_ids.shape}")

# Step 2: Mask a random position in each sequence
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
mask_token_id = tokenizer.mask_token_id
cls_token_id = tokenizer.cls_token_id
sep_token_id = tokenizer.sep_token_id
pad_token_id = tokenizer.pad_token_id

# Create attention masks
attention_masks = (input_ids != pad_token_id).long()

# Store original input_ids for later comparison

# Mask a random token in each sequence
batch_size = 1# Adjust based on your system's memory capacity

# dataset = TensorDataset(input_ids, attention_masks, labels)
# dataloader = DataLoader(dataset, batch_size=batch_size)

model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()  # Set the model to evaluation mode

# Optional: Move model and data to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


for batch_num in range(0, len(input_ids), batch_size):
    batch = input_ids[batch_num:batch_num + batch_size]
    batch = batch.to(device)
    
    metric_std, metric_std_r = evaluate_path_consistency(model, batch, num_permutations=10)
    print(f"Batch {batch_num // batch_size + 1}: {metric_std.mean().item():.4f} (mean), {metric_std_r.mean().item():.4f} (mean relative)")
    break

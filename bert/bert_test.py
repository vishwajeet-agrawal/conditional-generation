from transformers import BertTokenizer, BertForMaskedLM
import torch

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Input sentence with a mask token
sentence = "The quick brown fox jumps over the [MASK] dog."

# Tokenize the input
inputs = tokenizer(sentence, return_tensors="pt")
mask_token_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
print(inputs)
# Get predictions
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Find the most probable token
mask_token_logits = logits[0, mask_token_index, :]
top_token = torch.argmax(mask_token_logits, dim=1)

# Decode the predicted token
predicted_token = tokenizer.decode(top_token)

print(f"Predicted token: {predicted_token}")

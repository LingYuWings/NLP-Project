import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})  # Add pad token
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Define the dataset class
class ConversationDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        with open(file_path, "r") as file:
            for line in file:
                data = json.loads(line)
                text = data["text"]
                self.examples.append(text)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        text = self.examples[idx]
        
        # Truncate or adjust text length if needed
        if len(text) > self.max_length:
            text = text[:self.max_length]
        
        input_ids = self.tokenizer.encode(text, max_length=self.max_length, truncation=True, padding="max_length")
        return torch.tensor(input_ids)
        
# Define training parameters
batch_size = 4
learning_rate = 1e-4
num_epochs = 5

# Define file paths
data_dir = "./movie-corpus"
train_file_path = os.path.join(data_dir, "utterances.jsonl")

# Load and preprocess the dataset
train_dataset = ConversationDataset(train_file_path, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Prepare optimizer and loss function
optimizer = AdamW(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids = batch.to(model.device)
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]  # Shift labels by one position
        
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

# Save the fine-tuned model
save_path = "fine_tuned_gpt2"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("Fine-tuned model saved at", save_path)

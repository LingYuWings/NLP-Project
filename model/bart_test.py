import pandas as pd
import torch
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from torch.utils.data import DataLoader, Dataset

# 1. 数据预处理
class ConversationDataset(Dataset):
    def __init__(self, conversations, tokenizer, max_length):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conversation = self.conversations.iloc[idx]
        input_text = conversation['Speaker1'] + " <|endoftext|> " + conversation['Speaker2']
        inputs = self.tokenizer.encode(input_text, add_special_tokens=True, max_length=self.max_length, truncation=True)
        return {"input_ids": torch.tensor(inputs, dtype=torch.long)}

# Load and preprocess your CSV data
conversations_df = pd.read_csv("./formatted_csv.csv")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
max_length = 128
dataset = ConversationDataset(conversations_df, tokenizer, max_length)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 2. 模型选择和准备
model_config = BartConfig.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration(config=model_config)

# 3. 微调模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for batch in dataloader:
        inputs = batch['input_ids'].to(device)
        labels = batch['input_ids'].to(device)  # Auto-regressive decoding, use the same input as target
        optimizer.zero_grad()
        outputs = model(input_ids=inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {loss.item()}")

# 4. 保存模型
model.save_pretrained("chatbot_bart_model")

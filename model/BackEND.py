import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel

def tokenize_texts(texts,tokenizer,max_len=32):
    return tokenizer(texts, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')


class TextDataset(Dataset):
    def __init__(self, texts_arr, labels):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        tokenized_inputs=tokenize_texts(texts_arr,self.tokenizer)
        self.input_ids = tokenized_inputs['input_ids']
        self.attention_mask = tokenized_inputs['attention_mask']
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {'input_ids': self.input_ids[idx],'attention_mask': self.attention_mask[idx],'labels': self.labels[idx]}
    


class BERTLSTMClassifier(nn.Module):
    def __init__(self,hidden_dim, output_dim):
        super(BERTLSTMClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.lstm = nn.LSTM(self.bert.config.hidden_size, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        lstm_out, _ = self.lstm(bert_output.last_hidden_state)
        output = self.fc(lstm_out[:, -1, :])
        return output
    

def train(model,data,optimizer,criterion,epochs=5):
    model.train()
    for ep in range(epochs):
        for temp in data:
            input_ids = temp['input_ids']
            attention_mask = temp['attention_mask']
            labels = temp['labels']

            out=model(input_ids,attention_mask)

            loss=criterion(labels,out)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

        print(f"Epoch {ep + 1}/{epochs}, Loss: {loss.item()}")

        

















    




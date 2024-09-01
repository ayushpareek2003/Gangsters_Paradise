print("############ Welcome to Gangsters Paradise ############ \n")

print("                     Ayush Pareek ")
print("                     02-09-2024 \n \n")


import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset
import json
import warnings


warnings.filterwarnings("ignore") ## otherwise we will get alot of warning while loading our model 

path=str(input("Enter path to the trained model in torch format"))
model = torch.load(path)

print("Model Loaded \n")

print("Loading Bert tokenizer")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print("Bert tokenizer Loaded Successfully!.. \n")

print("Enter path for Mapping and Punishment Json files")
MP=str(input("Mapping"))
PP=str(input("Punishment \n"))

with open(MP, 'r') as file:
    MD = json.load(file)


with open(PP, 'r') as File:
    PD= json.load(File)



offense=str(input("Enter the Description of offense happened \n"))
off=[]
off.append(offense)
lab=[0]

### same class which was in backend ####
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }



train_dataset = TextDataset(off, lab, tokenizer)
train_dataset=train_dataset[0]
input_ids = train_dataset['input_ids'].unsqueeze(0)
attention_mask = train_dataset['attention_mask'].unsqueeze(0)

model.to(torch.device('cpu'))
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  

predicted_label = torch.argmax(logits, dim=-1).item()

print(MD[str(predicted_label)],"  is applicable in this situation and punishment for such crime is  ",PD[str(predicted_label)])





















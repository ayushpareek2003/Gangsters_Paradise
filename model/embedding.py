from transformers import BertTokenizer, BertModel
import torch


class embedding():
    def __init__(self):
        model_name = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)


    def forward(self,text_arr):
        tokenized_text_arr=self.tokenizer(text_arr, return_tensors='pt', truncation=True, padding=True, max_length=32)
        with torch.no_grad():
            outputs = self.model(**tokenized_text_arr)
            # Use the last hidden state as the embedding
            embeddings = outputs.last_hidden_state
        
        sentence_embeddings = torch.mean(embeddings, dim=1)
        return sentence_embeddings



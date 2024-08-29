from transformers import MarianTokenizer, MarianMTModel

class marian_loader():
    def __init__(self):
        self.languages=["en","fr","de","es","ar"] ##n


        self.model_right_flow=[]
        self.tokenizer_right_flow=[]
        self.model_left_flow=[]
        self.tokenizer_left_flow=[]

        self.model_right_individual=[]
        self.tokenizer_right_individual=[]
        self.model_left_individual=[]
        self.tokenizer_left_individual=[]

        for i in range(0,len(self.languages)-1):

            ##for chain conversion 
            self.model_right_flow.append(MarianMTModel.from_pretrained(f'Helsinki-NLP/opus-mt-{self.languages[i]}-{self.languages[i+1]}'))
            self.tokenizer_right_flow.append(MarianTokenizer.from_pretrained(f'Helsinki-NLP/opus-mt-{self.languages[i]}-{self.languages[i+1]}'))

            self.model_left_flow.append(MarianMTModel.from_pretrained(f'Helsinki-NLP/opus-mt-{self.languages[i+1]}-{self.languages[i]}'))
            self.tokenizer_left_flow.append(MarianTokenizer.from_pretrained(f'Helsinki-NLP/opus-mt-{self.languages[i+1]}-{self.languages[i]}'))



            ##individual model
            self.model_right_individual.append(MarianMTModel.from_pretrained(f'Helsinki-NLP/opus-mt-{"en"}-{self.languages[i+1]}'))
            self.tokenizer_right_individual.append(MarianTokenizer.from_pretrained(f'Helsinki-NLP/opus-mt-{"en"}-{self.languages[i+1]}'))

            self.model_left_individual.append(MarianMTModel.from_pretrained(f'Helsinki-NLP/opus-mt-{self.languages[i+1]}-{"en"}'))
            self.tokenizer_left_individual.append(MarianTokenizer.from_pretrained(f'Helsinki-NLP/opus-mt-{self.languages[i+1]}-{"en"}'))


    def translate(self,text, model, tokenizer):
  
        tokens = tokenizer(text, return_tensors="pt", padding=True)
        translated = model.generate(**tokens) 
        return tokenizer.decode(translated[0], skip_special_tokens=True)

    def forward(self,data):
        

        text=[]
        label=[]

        for i in range(len(data)):
            print(i)
            
            flow_forw=data['Offense'].iloc[i]

            for j in range(0,4,1):
                flow_forw=self.translate(flow_forw,self.model_right_flow[j],self.tokenizer_right_flow[j])
            for j in range(3,-1,-1):
                flow_forw=self.translate(flow_forw,self.model_left_flow[j],self.tokenizer_left_flow[j]) 

            text.append(flow_forw)
            label.append(data['Section'].iloc[i])
            
            flow_forw=data['Offense'].iloc[i]
            for j in range(0,4,1):
                flow_forw=self.translate(flow_forw,self.model_right_individual[j],self.tokenizer_right_individual[j])

                flow_forw=self.translate(flow_forw,self.model_left_individual[j],self.tokenizer_left_individual[j])

                text.append(flow_forw)
                label.append(data['Section'].iloc[i])

        return text,label



























        

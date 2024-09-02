## Gangsters Paradise

**Gangsters Paradise** is a machine learning model designed to predict the relevant Indian Penal Code (IPC) section based on user-provided descriptions.

### Requirements

To use this service, ensure you have the required Python packages installed. You can install the dependencies listed in `environment.txt` using:

```bash
pip install -r environment.txt
```

You will also need to download the following resources:
* Trained Model
* Reverse Mapping for IPC Sections and Punishments
  
These files are available in Release section and Dataset_AUG folder 

### How It Works
The initial dataset is from Kaggle and contained around 500 descriptions, each corresponding to a specific IPC section. However, this amount was insufficient for building a robust classifier. To address this, we employed text augmentation techniques, specifically back translation, to generate additional data

Back Translation involves translating text through multiple languages to produce paraphrased versions of the original text

For this project, the augmentation process included translations through the following language sequence:

[English → French → German → Spanish → Arabic → English]

Additionally, we used direct back translation between English and all other 4 language. This process generated approximately 5 samples per IPC section, resulting in a dataset of around 2,000 to 2,500 data points

For the classification task, we applied transfer learning with the BERT-base-uncased model, utilizing it for both classification and tokenization

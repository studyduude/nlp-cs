from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import json

# Catégories et textes
categories = ["Politics", "Health", "Finance", "Travel", "Food", "Education", "Environment", "Fashion", "Science", "Sports", "Technology", "Entertainment"]
with open('data/train.json') as f:
    data = json.load(f)

# Création d'une liste de phrases et de labels
texts = []
labels = []
for label, texts_list in data.items():
    for text in texts_list:
        texts.append(text)
        labels.append(categories.index(label))

# Préparation des données pour BERT
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Tokenisation des textes
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
labels = torch.tensor(labels)

# Création du dataset
dataset = Dataset(inputs, labels)

# Division en ensemble d'entraînement et de validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Entraînement de BERT
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(categories))

training_args = TrainingArguments(
    output_dir='./results',          
    num_train_epochs=200,              
    per_device_train_batch_size=8,  
    per_device_eval_batch_size=8,   
    warmup_steps=500,                
    weight_decay=0.01,               
    logging_dir='./logs',            
)

trainer = Trainer(
    model=model,                         
    args=training_args,                  
    train_dataset=train_dataset,         
    eval_dataset=val_dataset             
)

trainer.train()


with open('data/test_shuffle.txt') as f:
    test_texts = f.read().splitlines()

# Tokenisation des données de test
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")

# Prédiction
model.eval()
with torch.no_grad():
    outputs = model(**test_encodings)

predictions = torch.argmax(outputs.logits, dim=-1)

# Convertir les indices en catégories
predicted_categories = [categories[prediction] for prediction in predictions]

# Création du DataFrame
results_df = pd.DataFrame({
    'Category': predicted_categories,
    'Sentence': test_texts
})

# Sauvegarde en CSV
results_df.to_csv('predicted_test_categories.csv', index=False)

print("Les prédictions ont été sauvegardées dans 'predicted_test_categories.csv'")
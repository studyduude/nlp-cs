import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset, ConcatDataset
from sklearn.metrics import accuracy_score


label_list = ["Politics", "Health", "Finance", "Travel", "Food", "Education", "Environment", "Fashion", "Science", "Sports", "Technology", "Entertainment"]

# Load categories and data
def load_data(file_path):
    categories = ["Politics", "Health", "Finance", "Travel", "Food", "Education", "Environment", "Fashion", "Science", "Sports", "Technology", "Entertainment"]
    with open(file_path) as f:
        data = json.load(f)
    texts = []
    labels = []
    for label, texts_list in data.items():
        for text in texts_list:
            texts.append(text)
            labels.append(categories.index(label))
    return texts, labels, categories

def load_test_data(file_path):
    """Charge les données de test depuis un fichier texte, où chaque ligne est une phrase."""
    with open(file_path, 'r', encoding='utf-8') as file:
        test_texts = file.read().strip().split('\n')
    return test_texts

def tokenize_data(texts, tokenizer):
    # Reintroduce padding here to avoid the tensor size mismatch error
    return tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
    
# Create dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: self.encodings[key][idx].clone().detach() for key in self.encodings}
        # Assurez-vous que token_type_ids est présent
        if 'token_type_ids' not in item:
            item['token_type_ids'] = torch.zeros_like(item['input_ids'])
        
        if self.labels is not None:
            item['labels'] = self.labels[idx] if isinstance(self.labels[idx], torch.Tensor) else torch.tensor(self.labels[idx], dtype=torch.long)
        return item



# Training function
def train_model(model, train_dataset, val_dataset, epochs, tokenizer):
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,  # Adjust this as needed
        evaluation_strategy="steps",
        eval_steps=500,  # Adjust this as needed
        save_strategy="steps",  # Enable saving based on steps
        save_steps=500,  # Adjust this if needed to less frequent saves
        save_total_limit=1,  # Keep only the best model saved
        load_best_model_at_end=True,
        metric_for_best_model='loss'
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    )
    return trainer.train()





# Select uncertain samples
def select_uncertain_samples(model, dataset, num_samples, used_indices):
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    model.eval()
    softmax = torch.nn.Softmax(dim=1)
    uncertainties = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            outputs = model(**{k: v.to(model.device) for k, v in batch.items()})
            probabilities = softmax(outputs.logits)
            top_prob, top_cat = probabilities.max(dim=1)
            uncertainty = 1 - top_prob
            uncertainties.extend(uncertainty.tolist())

    # Sort uncertainties and filter out used indices
    uncertain_indices = np.argsort(uncertainties)[-num_samples:]
    filtered_indices = [idx for idx in uncertain_indices if idx not in used_indices]
    used_indices.update(filtered_indices)  # Update the set of used indices
    print(f"Top uncertainties: {sorted(uncertainties)[-10:]}")  # Shows the highest uncertainties
    print(f"Selected indices before filtering: {uncertain_indices[:10]}")  # Shows some of the indices selected for uncertainty
    print(f"Used indices count: {len(used_indices)}")  # Shows how many indices are already used


    return Subset(dataset, filtered_indices), used_indices

def prepare_for_custom_dataset(verified_samples):
    # Initialiser les dictionnaires pour contenir les valeurs accumulées
    encodings = {'input_ids': [], 'attention_mask': [], 'token_type_ids': []}
    labels = []

    # Collecter les valeurs des échantillons vérifiés
    for sample in verified_samples:
        # Assurer la présence de toutes les clés nécessaires, ou ignorer les échantillons incomplets
        for key in encodings.keys():
            if key in sample:
                encodings[key].append(sample[key])
            else:
                print(f"Missing '{key}' in verified samples, filling with zeros.")
                encodings[key].append(torch.zeros_like(sample['input_ids']))  # Utiliser des zéros si manquant

        labels.append(sample['label'])

    # Convertir les listes en tenseurs
    encodings = {key: torch.stack(vals) for key, vals in encodings.items()}
    labels = torch.tensor(labels, dtype=torch.long)

    return encodings, labels





def verify_labels(model, tokenizer, dataloader, label_list):
    verified_samples = []
    model.eval()

    for batch in dataloader:
        inputs = {k: v.to(model.device) for k, v in batch.items() if k != 'labels'}
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)

        for idx in range(inputs['input_ids'].size(0)):
            text = tokenizer.decode(inputs['input_ids'][idx], skip_special_tokens=True)
            predicted_label = label_list[predictions[idx].item()]

            print(f"Text: {text}")
            print(f"Predicted Label: {predicted_label}")

            user_input = input("Enter the correct label (press Enter if correct, type 'back' to redo the previous entry): ")
            if user_input == 'back':
                continue  # Handle 'back' action
            elif user_input in label_list or user_input == '':
                correct_label = user_input if user_input else predicted_label
                verified_samples.append({
                    'input_ids': inputs['input_ids'][idx],
                    'attention_mask': inputs['attention_mask'][idx],
                    'token_type_ids': inputs['token_type_ids'][idx] if 'token_type_ids' in inputs else torch.zeros_like(inputs['input_ids'][idx]),
                    'label': label_list.index(correct_label) if user_input else predictions[idx].item()
                })

    return verified_samples


# Active Learning Loop
def active_learning_loop(initial_train_dataset, val_dataset, test_dataset, model, tokenizer, num_iterations, num_samples_per_iter):
    used_indices = set()  # Ensemble pour suivre les indices déjà utilisés
    old_labels = save_label_states(initial_train_dataset)

    for iteration in range(num_iterations):
        print(f"--- Active Learning Iteration {iteration+1} ---")
        trainer = train_model(model, initial_train_dataset, val_dataset, epochs=700, tokenizer=tokenizer)
        
        uncertain_samples, used_indices = select_uncertain_samples(model, test_dataset, num_samples_per_iter, used_indices)
        
        samples_to_verify = DataLoader(uncertain_samples, batch_size=1, shuffle=False)
        verified_samples = verify_labels(model, tokenizer, samples_to_verify, label_list)

        if verified_samples:
            encodings, labels = prepare_for_custom_dataset(verified_samples)
            # Vérifiez si la liste des input_ids n'est pas vide en vérifiant la longueur
            if len(encodings['input_ids']) > 0:
                new_samples_dataset = CustomDataset(encodings, labels)
                initial_train_dataset = ConcatDataset([initial_train_dataset, new_samples_dataset])
                new_labels = save_label_states(initial_train_dataset)
                changed_labels = compare_label_states(old_labels, new_labels)
                if changed_labels:
                    print("Changed labels:")
                    for idx, (old, new) in changed_labels.items():
                        print(f"Sample {idx}: Old Label = {old}, New Label = {new}")
                old_labels = new_labels  # Mettre à jour les labels pour la prochaine comparaison
            else:
                print("No new data to add to the training set.")
        else:
            print("No verified samples to process.")
        save_predictions(model, test_dataset, tokenizer, iteration, num_samples_per_iter)

        print(f"--- Training completed for iteration {iteration+1} ---")

    return model


def save_label_states(dataset):
    # Cette fonction suppose que dataset est un objet de type CustomDataset qui contient les labels
    label_states = [sample['labels'].item() for sample in DataLoader(dataset, batch_size=1)]
    return label_states

def compare_label_states(old_labels, new_labels):
    changed_labels = {}
    for idx, (old, new) in enumerate(zip(old_labels, new_labels)):
        if old != new:
            changed_labels[idx] = (old, new)
    return changed_labels

def save_predictions(model, dataset, tokenizer, iteration, num_samples_per_iter, output_dir='./results'):
    """Sauvegarde les prédictions du modèle et compare avec les labels corrects."""
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    model.eval()
    predictions = []
    ids = []
    
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(**{k: v.to(model.device) for k, v in batch.items()})
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
    
    # Sauvegarde des prédictions dans un DataFrame
    df_preds = pd.DataFrame({'ID': range(0, len(predictions)), 'Label': [label_list[prediction] for prediction in predictions]})
    filename = f"{output_dir}/preds_iter{iteration}_samples{num_samples_per_iter}.csv"
    df_preds.to_csv(filename, index=False)
    
    # Chargement du fichier CSV de réponse et calcul de la précision
    df_true = pd.read_csv('data/baseline.csv')
    accuracy = accuracy_score(df_true['Label'], df_preds['Label'])
    print(f"Iteration {iteration}: Accuracy = {accuracy:.2f}")
    
    return accuracy

def main():
    texts, labels, categories = load_data('data/train.json')
    # Configuring the tokenizer to explicitly return or not return token_type_ids
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
    inputs = tokenizer(texts, return_token_type_ids=True, truncation=True, padding=True, return_tensors="pt")
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #inputs = tokenize_data(texts, tokenizer)
    dataset = CustomDataset(inputs, labels)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    test_texts = load_test_data('data/test_shuffle.txt')
    test_inputs = tokenize_data(test_texts, tokenizer)
    test_dataset = CustomDataset(test_inputs)  # No labels provided for test data

    print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(categories))
    final_model = active_learning_loop(train_dataset, val_dataset, test_dataset, model, tokenizer, num_iterations=5, num_samples_per_iter=10)


if __name__ == '__main__':
    main()


# from datasets import load_dataset
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support
#
#
# # 1. Carica il dataset
# print("Caricamento del dataset...")
# dataset = load_dataset('csv', data_files='training_dataset_corrected.csv')
#
# print("Nomi delle colonne nel dataset:", dataset['train'].column_names)
#
# # 2. Splitta il dataset in train/test
# print("Divisione del dataset in training e test...")
# dataset = dataset['train'].train_test_split(test_size=0.2)
#
# # (Opzionale) Usa un sottoinsieme per test rapidi
# # dataset = dataset.shuffle(seed=42).select(range(1000))
#
# # 3. Prepara il tokenizer
# print("Caricamento del tokenizer...")
# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
#
# def tokenize_function(examples):
#     # Combina titolo e descrizione in una singola stringa
#     combined_text = [f"{title} {desc}" for title, desc in zip(examples['Title'], examples['Description'])]
#     tokenized = tokenizer(combined_text, padding="max_length", truncation=True)
#     # Mappa le etichette da "Relevant" e "Not Relevant" a 1 e 0
#     label_map = {"Relevant": 1, "Not Relevant": 0}
#     tokenized["labels"] = [label_map[label] for label in examples["Relevance"]]
#     return tokenized
#
# # 4. Tokenizza il dataset
# print("Tokenizzazione del dataset...")
# tokenized_datasets = dataset.map(tokenize_function, batched=True)
#
# # 5. Carica il modello
# model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
#
# # 6. Definisci le metriche per la valutazione
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = logits.argmax(axis=-1)
#     acc = accuracy_score(labels, predictions)
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
#     return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}
#
# # 7. Configura gli argomenti di training
# training_args = TrainingArguments(
#     output_dir="./results",
#     evaluation_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=4,  # Era 8, diminuito a 4
#     per_device_eval_batch_size=4,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     logging_dir="./logs",
#     logging_steps=50,  # Mostra i log meno frequentemente
#     save_steps=500,
#     save_total_limit=2,
#     seed=42,
#     resume_from_checkpoint=True  # Per riprendere da un checkpoint
# )
#
# # 8. Inizializza il trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["test"],
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics
# )
#
# # 9. Addestra il modello
# print("Inizio dell'addestramento...")
# trainer.train()


from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Utilizzo del dispositivo: {device}")

# Carica il dataset
print("Caricamento del dataset...")
dataset = load_dataset('csv', data_files='training_dataset_corrected.csv')

print("Nomi delle colonne nel dataset:", dataset['train'].column_names)

# Splitta il dataset
print("Divisione del dataset in training e test...")
dataset = dataset['train'].train_test_split(test_size=0.2)

# Tokenizzazione
print("Caricamento del tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    combined_text = [f"{title} {desc}" for title, desc in zip(examples['Title'], examples['Description'])]
    tokenized = tokenizer(combined_text, padding="max_length", truncation=True)
    label_map = {"Relevant": 1, "Not Relevant": 0}
    tokenized["labels"] = [label_map[label] for label in examples["Relevance"]]
    return tokenized

print("Tokenizzazione del dataset...")
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Modello
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model.to(device)

# Metriche
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Argomenti di training
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # Sostituito evaluation_strategy
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    seed=42,
    fp16=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics  # Rimosso tokenizer
)

# Addestramento
print("Inizio dell'addestramento...")
trainer.train()

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch
import json
from sklearn.metrics import accuracy_score, classification_report

def load_and_prepare_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    df = df[['lemmatized_tweet', 'occupation']]
    
    label_dict = {label: i for i, label in enumerate(df['occupation'].unique())}
    df['label'] = df['occupation'].map(label_dict)
    
    print(f"Toplam meslek sayısı: {len(label_dict)}")
    print("\nMeslek dağılımı:")
    print(df['occupation'].value_counts())
    
    return df, label_dict

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}

def prepare_dataset(df, tokenizer, max_length=256):
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    def tokenize_function(examples):
        return tokenizer(examples['lemmatized_tweet'], 
                        padding='max_length',
                        truncation=True,
                        max_length=max_length)
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    return train_dataset, val_dataset

def train_model(train_dataset, val_dataset, num_labels, model_name="dbmdz/bert-base-turkish-cased"):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels
    )
    
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    print("Model eğitimi başlıyor...")
    trainer.train()
    
    return trainer, model

def evaluate_model(trainer, val_dataset, label_dict):
    predictions = trainer.predict(val_dataset)
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids
    
    reverse_label_dict = {v: k for k, v in label_dict.items()}
    
    true_occupations = [reverse_label_dict[label] for label in labels]
    predicted_occupations = [reverse_label_dict[pred] for pred in preds]
    
    report = classification_report(true_occupations, predicted_occupations)
    print("\nModel Değerlendirme Raporu:")
    print(report)

def main():
    print("Veri yükleniyor...")
    df, label_dict = load_and_prepare_data('cleaned-data-zemberek.json')
    
    print("\nModel ve tokenizer yükleniyor...")
    model_name = "dbmdz/bert-base-turkish-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("\nVeri setleri hazırlanıyor...")
    train_dataset, val_dataset = prepare_dataset(df, tokenizer)
    
    print("\nModel eğitimi başlıyor...")
    trainer, model = train_model(train_dataset, val_dataset, len(label_dict))
    
    print("\nModel değerlendirmesi yapılıyor...")
    evaluate_model(trainer, val_dataset, label_dict)
    
    print("\nModel kaydediliyor...")
    model.save_pretrained("./saved_model")
    tokenizer.save_pretrained("./saved_model")
    
    with open('./saved_model/label_dict.json', 'w', encoding='utf-8') as f:
        json.dump(label_dict, f, ensure_ascii=False, indent=4)
    
    print("\nİşlem tamamlandı!")

if __name__ == "__main__":
    main()
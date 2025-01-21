import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_prepare_data(json_path, n=1):
    # read the json file
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # take only the necessary columns
    df = df[['lemmatized_tweet', 'occupation']]

    # transform occupation names to labels
    label_dict = {label: i for i, label in enumerate(
        df['occupation'].unique())}
    df['label'] = df['occupation'].map(label_dict)

    # combine n tweets into one
    combined_tweets = []
    combined_labels = []

    for occupation, group in df.groupby('occupation'):
        tweets = group['lemmatized_tweet'].tolist()
        for i in range(0, len(tweets), n):
            combined_tweet = ' '.join(tweets[i:i + n])
            combined_tweets.append(combined_tweet)
            combined_labels.append(occupation)

    # new dataframe
    combined_df = pd.DataFrame(
        {'lemmatized_tweet': combined_tweets, 'occupation': combined_labels})
    combined_df['label'] = combined_df['occupation'].map(label_dict)

    print(f"Toplam meslek sayısı: {len(label_dict)}")
    print("\nMeslek dağılımı:")
    print(combined_df['occupation'].value_counts())

    return combined_df, label_dict


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}


def prepare_dataset(df, tokenizer, max_length=256):
    # split the data into training and validation sets
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label'])

    # Tokenization
    def tokenize_function(examples):
        return tokenizer(examples['lemmatized_tweet'],
                         padding='max_length',
                         truncation=True,
                         max_length=max_length)

    # Dataset
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    # tokenize the datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    return train_dataset, val_dataset


def train_model(train_dataset, val_dataset, num_labels, model_name="dbmdz/bert-base-turkish-cased"):
    # load the pretrained model and adjust the number of labels
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )

    # training arguments
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

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # Train the model
    print("Model eğitimi başlıyor...")
    trainer.train()

    return trainer, model


def evaluate_model(trainer, val_dataset, label_dict):
    # Make predictions on the validation set
    predictions = trainer.predict(val_dataset)
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids

    reverse_label_dict = {v: k for k, v in label_dict.items()}

    # True and predicted labels
    true_occupations = [reverse_label_dict[label] for label in labels]
    predicted_occupations = [reverse_label_dict[pred] for pred in preds]

    # classification report
    report = classification_report(
        true_occupations, predicted_occupations, output_dict=True)
    print("\nModel Değerlendirme Raporu:")
    print(json.dumps(report, indent=4, ensure_ascii=False))

    # save the report
    with open('./results/evaluation_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=4)

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=reverse_label_dict.values(
    ), yticklabels=reverse_label_dict.values())
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig('./results/confusion_matrix.png')
    plt.close()


def main():
    print("Veri yükleniyor...")
    n = 5  # n tweets will be combined into one

    # you can change the json file name in order to use a different dataset(with zemberek/zeyrek)
    df, label_dict = load_and_prepare_data('cleaned-data-zemberek.json', n=n)

    print("\nModel ve tokenizer yükleniyor.")
    model_name = "dbmdz/bert-base-turkish-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("\nVeri setleri hazırlanıyor.")
    train_dataset, val_dataset = prepare_dataset(df, tokenizer)

    print("\nModel eğitimi başlıyor.")
    trainer, model = train_model(train_dataset, val_dataset, len(label_dict))

    print("\nModel değerlendirmesi yapılıyor.")
    evaluate_model(trainer, val_dataset, label_dict)

    # save the model
    print("\nModel kaydediliyor...")
    model.save_pretrained("./saved_model")
    tokenizer.save_pretrained("./saved_model")

    # save the label dictionary
    with open('./saved_model/label_dict.json', 'w', encoding='utf-8') as f:
        json.dump(label_dict, f, ensure_ascii=False, indent=4)

    print("\nİşlem tamamlandı.")


if __name__ == "__main__":
    main()

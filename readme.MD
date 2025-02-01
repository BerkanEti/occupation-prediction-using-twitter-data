# Occupation Prediction Using Twitter Data

## Overview

This project aims to predict users' occupations based on their tweets using various machine learning (ML) and deep learning (DL) models. The dataset consists of 54,000 tweets in Turkish, collected through an API and enriched with additional data. The project implements multiple Natural Language Processing (NLP) techniques to preprocess and classify textual data efficiently.

## Features

- **Natural Language Processing (NLP)**: Implements preprocessing techniques such as tokenization, stop-word removal, and lemmatization.
- **Machine Learning Models**: Logistic Regression (LR), Multinomial Naive Bayes (MNB), Random Forest (RF), Support Vector Machines (SVM), and Gradient Boosting.
- **Deep Learning Models**: Multi-Layer Perceptron (MLP) and Convolutional Neural Networks (CNN).
- **Pre-trained Model**: The project leverages the Pre-trained Turkish BERT model for enhanced text classification accuracy.
- **Multi-Tweet Combinations**: Evaluates single, binary, triple, and quintuple tweet combinations to improve prediction accuracy.

## Dataset

The dataset comprises tweets categorized into 10 distinct occupations:

- Lawyer
- Dietitian
- Doctor
- Economist
- Psychologist
- Sports Commentator
- Historian
- Software Developer
- Agricultural Engineer
- Teacher

Data Sources:

- **Mayda dataset** (43,000 tweets)
- **Newly collected data** (11,000 tweets via API)

## Data Preprocessing

- **Text Cleaning**: Removing URLs, mentions, emojis, and punctuation.
- **Lemmatization**: Applied using **Zeyrek** and **Zemberek** NLP libraries.
- **Stop-word Removal**: Eliminated unnecessary Turkish words using NLTK.
- **Data Combination**: Tweets were grouped into single, binary, triple, and quintuple combinations to enhance contextual understanding.

## Model Performance

| Model                   | NLP Library | Single | Binary | Triple | Quintuple |
| ----------------------- | ----------- | ------ | ------ | ------ | --------- |
| Logistic Regression     | Zeyrek      | 72.7%  | 83.9%  | 90.7%  | 94.2%     |
| Logistic Regression     | Zemberek    | 72.1%  | 84.4%  | 90.5%  | 94.6%     |
| Random Forest           | Zeyrek      | 66.4%  | 76.4%  | 82.8%  | 87.9%     |
| Random Forest           | Zemberek    | 65.9%  | 76.0%  | 83.0%  | 88.7%     |
| Support Vector Machines | Zeyrek      | 74.2%  | 84.6%  | 91.2%  | 94.6%     |
| Support Vector Machines | Zemberek    | 73.5%  | 85.0%  | 91.1%  | **95.1%** |
| Multinomial Naive Bayes | Zeyrek      | 72.1%  | 82.9%  | 89.5%  | 92.7%     |
| Multinomial Naive Bayes | Zemberek    | 71.3%  | 82.6%  | 89.6%  | 92.7%     |
| Gradient Boosting       | Zeyrek      | 64.9%  | 76.2%  | 82.7%  | 86.4%     |
| Gradient Boosting       | Zemberek    | 64.7%  | 75.5%  | 82.7%  | 86.2%     |
| CNN                     | Zeyrek      | 70.5%  | 81.0%  | 87.3%  | 93.1%     |
| CNN                     | Zemberek    | 69.8%  | 81.2%  | 86.7%  | 92.4%     |
| MLP                     | Zeyrek      | 73.5%  | 83.0%  | 89.8%  | 94.5%     |
| MLP                     | Zemberek    | 71.9%  | 83.7%  | 88.4%  | 93.3%     |
| Pre-trained BERT        | Zeyrek      | 78.0%  | 86.8%  | 92.0%  | 93.4%     |
| Pre-trained BERT        | Zemberek    | 76.0%  | 86.4%  | 90.6%  | 94.3%     |

### Best Performing Model

The highest accuracy was achieved using **Support Vector Machines (SVM) with Zemberek preprocessing** on **quintuple tweet combinations**, reaching **95.1% accuracy**.

## Future Work

- Expanding the dataset with real-time Twitter data.
- Incorporating **Word2Vec** and **GloVe** embeddings.
- Exploring large language models (LLMs) for enhanced classification.

## Contributors

- **Berkan Eti** ([GitHub](https://github.com/berkaneti))
- **Ömer Diner** ([GitHub](https://github.com/omerdiner))

## License

© 2025 Yildiz Technical University. All rights reserved.

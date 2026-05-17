# Part 3: NLP and Sequence Modeling Mini Project

## Overview

This repository contains the solution for **Part 3** of the assignment. The goal is to build a complete NLP pipeline to classify customer support messages by sentiment, and compare traditional text vectorization with sequence-based deep learning.

---

## Dataset

- **File:** `customer_support_text_classification.csv`
- **Source:** [Google Drive – Part 3 Dataset](https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing)
- **Records:** 1,500 | **Target:** `sentiment_label` (positive / neutral / negative)

> ⚠️ Dataset is not uploaded to this repository. Download it from the link above and place it in the same folder as the notebook before running.

---

## Approach & Steps

### Task 1 – Dataset Understanding
- Loaded dataset and explored shape, class distribution, sample messages
- Analyzed average text length and word count per sentiment class
- Visualized class balance and text length distribution

### Task 2 – Text Preprocessing
Full cleaning pipeline applied to `customer_message`:
- **Lowercasing** → uniform text
- **Special character removal** → strip punctuation, numbers, symbols
- **Tokenization** → split into individual words
- **Stopword removal** → remove common uninformative words (NLTK English stopwords)
- **Lemmatization** → reduce words to base form (e.g., "running" → "run")

### Task 3 – Text Vectorization

**Why text must be vectorized:** ML models work with numbers. Text must be converted to numerical vectors so mathematical operations (gradients, dot products, distance calculations) can be performed.

| Method | Description |
|---|---|
| Bag of Words | Counts word occurrences; simple but ignores order |
| TF-IDF | Weights words by frequency × inverse document frequency; rewards rare but important words |
| Tokenizer Sequences | Integer-encoded padded sequences for LSTM input |

### Task 4 – Baseline Models

| Model | Vectorization | Accuracy |
|---|---|---|
| Naive Bayes | Bag of Words | ~85% |
| Logistic Regression | TF-IDF | ~87% |

### Task 5 – Sequence Model: Bidirectional LSTM

**Architecture:**
```
Input Sequence (max_len=50)
  → Embedding(vocab=10000, dim=64)
  → Bidirectional LSTM(64, return_sequences=True) + Dropout(0.3)
  → LSTM(32) + Dropout(0.3)
  → Dense(64, ReLU)
  → Dense(3, Softmax)
```
- **Loss:** Sparse Categorical Cross-Entropy
- **Optimizer:** Adam
- **Callbacks:** EarlyStopping (patience=5)

### Task 6 – Attention and Transformer Reflection

**RNN limitation:** Fixed hidden state causes vanishing gradients over long sequences — earlier information is forgotten.

**LSTM advantage:** Three gating mechanisms (forget, input, output) + a cell state highway allow selective long-term memory.

**Attention:** Allows the model to directly reference any position in the input when generating output — eliminates the information bottleneck of fixed hidden states.

**Transformers:** Replace recurrence with parallel self-attention. Every token attends to every other token from layer 1. This enables massive scaling (GPT, BERT, Claude) and powers modern NLP and Generative AI.

---

## Repository Structure

```
part-3-nlp-sequence-modeling/
│
├── README.md
├── notebook.ipynb
├── requirements.txt
└── results/
    ├── dataset_analysis.png
    ├── baseline_confusion_matrices.png
    ├── lstm_training_curves.png
    ├── model_evaluation.png
    ├── model_evaluation.csv
    └── sample_predictions.txt
```

---

## How to Run

1. Upload `notebook.ipynb` to [Google Colab](https://colab.research.google.com)
2. Upload `customer_support_text_classification.csv` via the 📁 sidebar
3. Click **Runtime → Run All**
4. Download outputs from `results/` folder

---

## Requirements

See `requirements.txt`. Key libraries: `tensorflow`, `nltk`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`

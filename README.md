# MLP_Final_Project_Sentiment_analysis

# 📊 Financial Sentiment Analysis Using NLP & ML

This project explores **financial sentiment analysis** using two different datasets and machine learning models to predict stock movements and classify sentiment in financial texts.

---

## Overview

This repo contains two main parts:

1. **Stock Market News Sentiment Prediction**
   - Dataset: [`Combined_News_DJIA.csv`](https://www.kaggle.com/datasets/aaron7sun/stocknews)
   - Goal: Predict whether the Dow Jones Industrial Average (DJIA) index will go up or down based on daily news headlines.
   - Approach: News preprocessing → Bag-of-Words (bi-gram) → Logistic Regression, Random Forest, Naive Bayes, MLP Classifier.

2. **Financial Phrase Sentiment Classification**
   - Dataset: [`FinancialPhraseBank v1.0`](https://www.researchgate.net/publication/251231364_FinancialPhraseBank-v10)
   - Goal: Classify short financial sentences into **positive**, **neutral**, or **negative** sentiments.
   - Approach: Text cleaning → Label encoding → TF-IDF → Classifiers like Logistic Regression, Random Forest, Naive Bayes, MLP Classifier.

---

## Models Used

- Logistic Regression  
- Random Forest  
- Multinomial Naive Bayes  
- MLP Classifier (Neural Net)

---

## Datasets

| Dataset | Source | Description |
|--------|--------|-------------|
| `Combined_News_DJIA.csv` | [Kaggle](https://www.kaggle.com/datasets/aaron7sun/stocknews) | 25 news headlines per day with DJIA movement label |
| `FinancialPhraseBank-v1.0` | [ResearchGate](https://www.researchgate.net/publication/251231364_FinancialPhraseBank-v10) | Labeled financial phrases with sentiment labels |

---

## performance Summary 

### For Financial Phrase Sentiment Classification
| Model          | Accuracy | Precision (Macro Avg) | Recall (Macro Avg) | F1-Score (Macro Avg) |
|----------------|----------|----------------------|--------------------|---------------------|
| SVM            | 82.6%    | 0.83                 | 0.74               | 0.77                |
| Naive Bayes    | 77.1%    | 0.81                 | 0.55               | 0.53                |
| Random Forest  | 82.4%    | 0.81                 | 0.71               | 0.74                |
| MLP Classifier | 81.8%    | 0.76                 | 0.75               | 0.76                |

# For Stock Market News Sentiment Prediction to predict the stock going up or down 
| Model            | Accuracy | Precision | Recall | F1 Score |
|------------------|----------|-----------|--------|----------|
| Logistic Regression | 53.97%  | 0.54    | 0.63   | 0.58     |
| Random Forest      | 51.59%  | 0.51     | 0.83   | 0.63     |
| Naive Bayes        | 46.56%  | 0.47     | 0.45   | 0.46     |
| MLP Classifier     | 52.91%  | 0.53     | 0.63   | 0.58     |

---

## PreProcessing Highlights

- Stopword removal
- Stemming with `PorterStemmer`
- Bag-of-Words and TF-IDF vectorization
- Bi-grams used for headline sentiment modeling

--- 

## How to Run

```bash
# Install requirements
pip install -r requirements.txt

#Run the code 
python *.py
```
import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch

# Configuration
INPUT_DIR = "split_docs/"
OUTPUT_CSV = "sentiment_emotion_metrics.csv"

analyzer = SentimentIntensityAnalyzer()

# Models
sentiment_tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
sentiment_pipeline = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer)

emotion_tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
emotion_model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
emotion_pipeline = pipeline("text-classification", model=emotion_model, tokenizer=emotion_tokenizer, return_all_scores=False)

# Initialization
all_document_metrics = []

all_filenames = [f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")]

for filename in all_filenames:
    print(f"Processing {filename}...")
    filepath = os.path.join(INPUT_DIR, filename)

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    num_paragraphs = len(paragraphs)

    document_data = {"document": filename, "total_paragraphs": num_paragraphs}

# --------- Sentiment (VADER) ---------
# -------------------------------------

    vader_compound = []
    vader_pos = 0
    vader_neg = 0
    vader_neu = 0

    for para in paragraphs:
        vs = analyzer.polarity_scores(para)
        vader_compound.append(vs['compound'])
        if vs['compound'] >= 0.05:
            vader_pos += 1
        elif vs['compound'] <= -0.05:
            vader_neg += 1
        else:
            vader_neu += 1

    document_data["avg_VADER_compound"] = sum(vader_compound) / num_paragraphs if num_paragraphs > 0 else 0
    document_data["VADER_positive"] = (vader_pos / num_paragraphs) * 100 if num_paragraphs > 0 else 0
    document_data["VADER_negative"] = (vader_neg / num_paragraphs) * 100 if num_paragraphs > 0 else 0
    document_data["VADER_neutral"] = (vader_neu / num_paragraphs) * 100 if num_paragraphs > 0 else 0

# -------- Sentiment (roBERTa) --------
# -------------------------------------

    roberta_sent_pos = 0
    roberta_sent_neg = 0
    roberta_sent_neu = 0

    for para in paragraphs:
        result = sentiment_pipeline(para[:512])[0]
        label = result['label']
        if label == 'positive':
            roberta_sent_pos += 1
        elif label == 'negative':
            roberta_sent_neg += 1
        elif label == 'neutral':
            roberta_sent_neu += 1

    document_data["roberta_positive"] = (roberta_sent_pos / num_paragraphs) * 100 if num_paragraphs > 0 else 0
    document_data["roberta_negative"] = (roberta_sent_neg / num_paragraphs) * 100 if num_paragraphs > 0 else 0
    document_data["roberta_neutral"] = (roberta_sent_neu / num_paragraphs) * 100 if num_paragraphs > 0 else 0

# --------- Emotion (roBERTa) ---------
# -------------------------------------

    emotion_labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    roberta_emotion_counts = {label: 0 for label in emotion_labels}

    for para in paragraphs:
        result = emotion_pipeline(para[:512])[0] # Truncate
        label = result['label']
        if label in roberta_emotion_counts:
            roberta_emotion_counts[label] += 1

    for label in emotion_labels:
        perc_emotion = (roberta_emotion_counts[label] / num_paragraphs) * 100 if num_paragraphs > 0 else 0
        document_data[f"emotion_{label}"] = perc_emotion

# DF conversion and saving as CSV

    all_document_metrics.append(document_data)
    print(f"Finished processing {filename}.")

df_combined = pd.DataFrame(all_document_metrics)

column_order = ["document", "total_paragraphs",
                "avg_VADER_compound", "VADER_positive", "VADER_negative", "VADER_neutral",
                "roberta_positive", "roberta_negative", "roberta_neutral"]
emotion_cols_ordered = [f"emotion_{label}" for label in ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]]
column_order.extend(emotion_cols_ordered)

existing_columns_in_order = [col for col in column_order if col in df_combined.columns]
df_combined = df_combined[existing_columns_in_order]


df_combined.to_csv(OUTPUT_CSV, index=False)
print(f"Done. Metrics saved to {OUTPUT_CSV}")
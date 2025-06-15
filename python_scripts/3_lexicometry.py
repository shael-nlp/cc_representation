import spacy
from spacy.tokens import DocBin
from sklearn.feature_extraction.text import TfidfVectorizer
import textstat
from collections import Counter
import math
import os
import pandas as pd

# Configuration
INPUT_DIR = "processed_docs/"
MODEL = "en_core_web_lg"
OUTPUT_CSV = "corpus_metrics.csv"

# Loading spaCy to use en_core_web_lg's stop words list
nlp = spacy.load(MODEL)
STOP_WORDS = nlp.Defaults.stop_words
print(f"Successfully loaded spaCy model and {len(STOP_WORDS)} stop words.")

all_doc_metrics_data = []
corpus_for_tfidf = []
doc_names = []

doc_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".spacy")]

for file_name in sorted(doc_files): # Sort for consistent order
    print(f"Now processing: {file_name} !")
    file_path = os.path.join(INPUT_DIR, file_name)

    doc_bin_loaded = DocBin().from_disk(file_path)
    loaded_docs_from_file = list(doc_bin_loaded.get_docs(nlp.vocab))

    # While we could have saved multiple documents per binary file
    # We decided to save only 1 document per file to keep document names
    doc = loaded_docs_from_file[0]

    doc_names.append(file_name)

    current_doc_metrics = {"document_name": file_name}

    raw_text = doc.text
    all_tokens = [token for token in doc if not token.is_space]
    words_no_punct = [token for token in all_tokens if not token.is_punct]
    words_no_punct_no_stop = [token for token in words_no_punct if token.text.lower() not in STOP_WORDS]
    lemmas_no_punct_no_stop = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space and not token.is_stop]

    # Word Counts
    current_doc_metrics["total_tokens_incl_punct"] = len(all_tokens)
    current_doc_metrics["total_words_excl_punct"] = len(words_no_punct)
    current_doc_metrics["total_words_excl_punct_stop"] = len(words_no_punct_no_stop)

    # Lexical Diversity
    total_lemmas_ld = len(lemmas_no_punct_no_stop)
    unique_lemmas_ld = len(set(lemmas_no_punct_no_stop))
    current_doc_metrics["TTR"] = (unique_lemmas_ld / total_lemmas_ld)
    current_doc_metrics["herdan_c"] = (math.log(unique_lemmas_ld) / math.log(total_lemmas_ld))
    current_doc_metrics["guiraud_r"] = (unique_lemmas_ld / math.sqrt(total_lemmas_ld))

    # Lexical Density
    content_pos = {'NOUN', 'VERB', 'ADJ', 'ADV', 'NUM', 'PROPN'}
    words_for_ld = words_no_punct
    content_word_tokens_for_ld = [token for token in words_for_ld if token.pos_ in content_pos]
    current_doc_metrics["lexical_density"] = (len(content_word_tokens_for_ld) / len(words_for_ld))

    # Readability
    current_doc_metrics["FRE"] = textstat.flesch_reading_ease(raw_text)
    current_doc_metrics["FKGL"] = textstat.flesch_kincaid_grade(raw_text)

    # Avg Sentence Length
    num_sentences = len(list(doc.sents))
    current_doc_metrics["num_sentences"] = num_sentences
    current_doc_metrics["avg_sentence_length"] = (current_doc_metrics["total_words_excl_punct"] / num_sentences)

    # Avg Word Length
    total_chars_in_words = sum(len(token.text) for token in words_no_punct)
    current_doc_metrics["avg_word_length"] = (total_chars_in_words / len(words_no_punct))

    # Relative Frequency of Function Words
    function_pos_categories = ['ADP','AUX','CCONJ','SCONJ','DET','PRON','PART', 'INTJ']
    function_word_count = sum(token.pos_ in function_pos_categories for token in words_no_punct)

    current_doc_metrics["rel_freq_function_words"] = (function_word_count / current_doc_metrics["total_words_excl_punct"]) * 100

    # POS Tags Distribution
    pos_tags_list = [token.pos_ for token in words_no_punct]
    pos_counts = Counter(pos_tags_list)
    total_words_for_pos = len(pos_tags_list)
    major_pos_cats = ['NOUN', 'VERB', 'AUX', 'ADJ', 'ADV', 'PRON', 'ADP', 'CCONJ', 'SCONJ']

    for pos_cat in major_pos_cats:
        current_doc_metrics[f"pos_{pos_cat}"] = (pos_counts.get(pos_cat, 0) / total_words_for_pos) * 100
    current_doc_metrics["pos_OTHER"] = sum(count for tag, count in pos_counts.items() if tag not in major_pos_cats and tag != 'SPACE') / total_words_for_pos * 100

    all_doc_metrics_data.append(current_doc_metrics)

    # Appending text for TF-IDF
    tfidf_text = " ".join([token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space and not token.is_stop])
    corpus_for_tfidf.append(tfidf_text)

# Configuration TF-IDF
vectorizer = TfidfVectorizer(max_features=2000)
tfidf_matrix = vectorizer.fit_transform(corpus_for_tfidf)
feature_names = vectorizer.get_feature_names_out()
num_top_tfidf_terms = 10

for i, doc_name_from_order in enumerate(doc_names):
    metrics_dict_for_doc = all_doc_metrics_data[i]

    doc_tfidf_scores = tfidf_matrix[i].toarray().flatten()
    top_indices = doc_tfidf_scores.argsort()[-num_top_tfidf_terms:][::-1]
    top_terms_scores = [(feature_names[j], doc_tfidf_scores[j]) for j in top_indices if doc_tfidf_scores[j] > 0.0001]
    metrics_dict_for_doc[f"top_{num_top_tfidf_terms}_tfidf_terms"] = "; ".join([f"{term}:{score:.4f}" for term, score in top_terms_scores])

# Conversion to DF and saving as CSV for analysis
metrics_df = pd.DataFrame(all_doc_metrics_data)
metrics_df.to_csv(OUTPUT_CSV, index=False)
print(f"Metrics successfully saved to: {OUTPUT_CSV}")
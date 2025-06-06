import spacy
from spacy.tokens import DocBin
from collections import Counter
import os
import re
import pandas as pd

# Configuration
INPUT_DIR = "processed_docs/"
OUTPUT_CSV = "modality_metrics.csv"
NORMALIZATION_FACTOR = 1000

MODAL_VERBS = [
    "can", "could", "may", "might", "must", "shall", "should", "will", "would"
]
MODAL_ADVERBS = ["apparently", "arguably", "assuredly", "certainly", "clearly",
    "conceivably", "definitely", "doubtless", "evidently", "hopefully",
    "indubitably", "ineluctably", "inescapably", "incontestably", "likely",
    "manifestly", "maybe", "necessarily", "obviously", "patently", "perhaps",
    "plainly", "possibly", "presumably", "probably", "seemingly", "surely",
    "truly", "unarguably", "unavoidably", "undeniably", "undoubtedly",
    "unquestionably"
]

# Confidence and Likelihood metrics are adapted from AR4:
# "Box 1.1: Treatment of Uncertainties in the Working Group I Assessment"

CONFIDENCE_DICT = {
    "very high confidence": "90-100%", "high confidence": "80%",
    "medium confidence": "50%", "low confidence": "20%",
    "very low confidence": "0-10%"
}

LIKELIHOOD_DICT = {
    "virtually certain": "99-100%", "extremely likely": "95-100%",
    "very likely": "90-100%", "likely": "66-100%",
    "more likely than not": "50-100%", "about as likely as not": "33-66%",
    "unlikely": "0-33%", "very unlikely": "0-10%",
    "extremely unlikely": "0-5%", "exceptionally unlikely": "0-1%"
}

nlp = spacy.load("en_core_web_lg")
all_results_data = []

doc_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".spacy")]

for filename in sorted(doc_files):
    file_path = os.path.join(INPUT_DIR, filename)
    doc_name = os.path.basename(filename)

    print(f"Currently processing: {doc_name} !")

    doc_bin_loaded = DocBin().from_disk(file_path)

    loaded_docs_from_file = list(doc_bin_loaded.get_docs(nlp.vocab))
    doc = loaded_docs_from_file[0]

    current_likelihood_counts = Counter()
    current_confidence_counts = Counter()
    current_modal_verb_counts = Counter()
    current_modal_adverb_counts = Counter()
    current_negation_modal_count = 0
    current_total_words = 0

    text_content_from_doc = doc.text
    lower_text_content = text_content_from_doc.lower()

    for phrase in LIKELIHOOD_DICT.keys():
        pattern = r'\b' + re.escape(phrase.lower()) + r'\b'
        matches = re.findall(pattern, lower_text_content)
        if matches:
            current_likelihood_counts[phrase] += len(matches)

    for phrase in CONFIDENCE_DICT.keys():
        pattern = r'\b' + re.escape(phrase.lower()) + r'\b'
        matches = re.findall(pattern, lower_text_content)
        if matches:
            current_confidence_counts[phrase] += len(matches)

    for token in doc:
        if not token.is_punct and not token.is_stop:
            current_total_words += 1

        if token.pos_ == "AUX" and token.lemma_ in MODAL_VERBS:
            current_modal_verb_counts[token.lemma_] += 1
            # Negation check
            if token.i + 1 < len(doc) and doc[token.i + 1].lemma_ == "not":
                current_negation_modal_count += 1
            elif token.head.lemma_ == "not" and token.head.i == token.i -1 :
                 current_negation_modal_count += 1

        if token.pos_ == "ADV" and token.lemma_ in MODAL_ADVERBS:
            current_modal_adverb_counts[token.lemma_] += 1

    for term, count in current_likelihood_counts.items():
        norm_freq = (count / current_total_words) * NORMALIZATION_FACTOR
        all_results_data.append([doc_name, "Likelihood", term, count, norm_freq, current_total_words])
    for term, count in current_confidence_counts.items():
        norm_freq = (count / current_total_words) * NORMALIZATION_FACTOR
        all_results_data.append([doc_name, "Confidence", term, count, norm_freq, current_total_words])
    for term, count in current_modal_verb_counts.items():
        norm_freq = (count / current_total_words) * NORMALIZATION_FACTOR
        all_results_data.append([doc_name, "Modal Verbs", term, count, norm_freq, current_total_words])
    for term, count in current_modal_adverb_counts.items():
        norm_freq = (count / current_total_words) * NORMALIZATION_FACTOR
        all_results_data.append([doc_name, "Modal Adverbs", term, count, norm_freq, current_total_words])
    norm_freq_neg = (current_negation_modal_count / current_total_words) * NORMALIZATION_FACTOR
    all_results_data.append([doc_name, "Negation Near Modal", "count", current_negation_modal_count, norm_freq_neg, current_total_words])

# --- Create Pandas DataFrame and Print/Save ---
metrics_df = pd.DataFrame(all_results_data, columns=[
    "Document", "Modal_Category", "Term", "Raw_Count",
    f"Normalized_Freq_{NORMALIZATION_FACTOR}", "Total_Words"
])

metrics_df.to_csv(OUTPUT_CSV, index=False)
print(f"Metrics successfully saved to: {OUTPUT_CSV}")
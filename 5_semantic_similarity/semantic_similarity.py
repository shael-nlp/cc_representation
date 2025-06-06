import spacy
from spacy.tokens import DocBin
from sentence_transformers import SentenceTransformer, util
import os
import numpy as np
import csv
import torch

# Configuration
INPUT_DIR = "processed_docs/"
OUTPUT_CSV = "sentence_similarity_metrics.csv"

DOC_PAIRS = [
     (os.path.join(INPUT_DIR, "AR3_WG3_SPM.spacy"), os.path.join(INPUT_DIR, "Wiki_CCM_2005-06-26.spacy")),
     (os.path.join(INPUT_DIR, "AR4_WG3_SPM.spacy"), os.path.join(INPUT_DIR, "Wiki_CCM_2008-05-07.spacy")),
     (os.path.join(INPUT_DIR, "AR5_WG3_SPM.spacy"), os.path.join(INPUT_DIR, "Wiki_CCM_2014-09-02.spacy")),
     (os.path.join(INPUT_DIR, "AR6_WG3_SPM.spacy"), os.path.join(INPUT_DIR, "Wiki_CCM_2022-06-13.spacy")),
]

SBERT_MODEL_NAME = 'all-MiniLM-L6-v2'

nlp = spacy.load("en_core_web_lg")
sbert = SentenceTransformer(SBERT_MODEL_NAME)

sent_sim_results = []
header = [
    "IPCC_Document",
    "Wikipedia_Document",
    "IPCC_Sentences_Count",
    "Wikipedia_Sentences_Count",
    "Mean_Similarity",
    "Median_Similarity"
]
sent_sim_results.append(header)


for filepath_a, filepath_b in DOC_PAIRS:
    filename_a = os.path.basename(filepath_a)
    filename_b = os.path.basename(filepath_b)
    print(f"Comparing Sentences from: {filename_a} and {filename_b}")

    doc_bin_a = DocBin().from_disk(filepath_a)
    doc_a = list(doc_bin_a.get_docs(nlp.vocab))[0]

    doc_bin_b = DocBin().from_disk(filepath_b)
    doc_b = list(doc_bin_b.get_docs(nlp.vocab))[0]

    sentences_a = [sent.text.strip() for sent in doc_a.sents if sent.text.strip()]
    sentences_b = [sent.text.strip() for sent in doc_b.sents if sent.text.strip()]

    print(f"Found {len(sentences_a)} sentences in {filename_a}, {len(sentences_b)} in {filename_b}.\nStarting embeddings generation.")

    embeddings_a = sbert.encode(sentences_a, convert_to_tensor=True, show_progress_bar=False)
    embeddings_b = sbert.encode(sentences_b, convert_to_tensor=True, show_progress_bar=False)

    print("Embeddings done. Calculating cosine similarity...")
    embeddings_b = embeddings_b.to(embeddings_a.device)
    cosine_scores_ab = util.cos_sim(embeddings_a, embeddings_b)

    scores = []
    for i in range(len(sentences_a)):
        best_match = torch.max(cosine_scores_ab[i]).item()
        scores.append(best_match)

    mean = f"{np.mean(scores):.3f}"
    median = f"{np.median(scores):.3f}"
    print(f"Mean: {mean}")
    print(f"Median: {median}")

    results_row = [
        filename_a,
        filename_b,
        len(sentences_a),
        len(sentences_b),
        mean,
        median
    ]
    sent_sim_results.append(results_row)

with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(sent_sim_results)
print(f"Done. Results saved to {OUTPUT_CSV}")
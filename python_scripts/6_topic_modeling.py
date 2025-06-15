import spacy
from spacy.tokens import DocBin
import glob
import os
from bertopic import BERTopic
import pandas as pd

# Configuration
INPUT_DIR = "processed_docs/"
OUTPUT_CSV = "topic_modeling_results.csv"

SPACY_MODEL = "en_core_web_lg"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

nlp = spacy.load(SPACY_MODEL)

corpus = []
doc_names = []

files = glob.glob(os.path.join(INPUT_DIR, "*.spacy"))

for filepath in files:
    doc_bin = DocBin().from_disk(filepath)
    docs = list(doc_bin.get_docs(nlp.vocab))
    
    doc = docs[0]
        
    processed_tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
        ]

    corpus.append(" ".join(processed_tokens))
        
    doc_names.append(os.path.splitext(os.path.basename(filepath))[0])

# Topic modeling config
topic_model = BERTopic(
    embedding_model=EMBEDDING_MODEL,
    language="english",
    nr_topics="auto", # Topic number set to "auto"
    min_topic_size=2, # Keep to 2, higher values might affect results quality (due to corpus size)
    verbose=True
    )

topics, probabilities = topic_model.fit_transform(corpus)

topic_info_df = topic_model.get_topic_info()

document_info_list = []
for i, doc_name in enumerate(doc_names):
    document_info_list.append({
        "Document_Name": doc_name,
        "Assigned_Topic_ID": topics[i],
        })

doc_topics = pd.DataFrame(document_info_list)

results_df = pd.merge(
    doc_topics,
    topic_info_df[['Topic', 'Name', 'Representation']],
    left_on='Assigned_Topic_ID',
    right_on='Topic',
    how='left'
    )

results_df.to_csv(OUTPUT_CSV, index=False)

print(f"Done. Results saved to: {OUTPUT_CSV}")
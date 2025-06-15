import os
import spacy
from spacy.tokens import DocBin
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, logging as hf_logging
import pandas as pd
from collections import Counter
import torch

# Configuration
INPUT_DIR_SPACY = "processed_docs/"
SPLIT_TEXT_DIR = "split_texts/"
OUTPUT_CSV = "results_ner.csv"

MODEL_CC = "nicolauduran45/specter-climate-change-NER"


nlp = spacy.load("en_core_web_lg")

tokenizer_hf = AutoTokenizer.from_pretrained(MODEL_CC)
model_hf = AutoModelForTokenClassification.from_pretrained(MODEL_CC)

# Config for CC NER
cc_ner_pipeline = pipeline(
    "ner",
    model=model_hf,
    tokenizer=tokenizer_hf,
    aggregation_strategy="simple",
    device=0
)

all_results = []

doc_files = [f for f in os.listdir(INPUT_DIR_SPACY) if f.endswith(".spacy")]

# Main loop
for filename in sorted(doc_files):
    filepath = os.path.join(INPUT_DIR_SPACY, filename)
    doc_name = os.path.basename(filename)

    # ----------------------- Part 1 -----------------------
    # ------------------------------------------------------
    print(f"Processing: {doc_name}")

    doc_bin = DocBin().from_disk(filepath)
    loaded_docs = list(doc_bin.get_docs(nlp.vocab))
    
    doc_obj = loaded_docs[0]

    # Total words from spaCy will be used for split_texts paragraphs as well
    # No need to recount words as texts are the same
    total_words = 0
    for token in doc_obj:
        if not token.is_punct and not token.is_stop:
            total_words += 1
    
    result_row = {'Document': doc_name, 'Total Words': total_words}

    # Labels were preprocessed in section 3.2
    spacy_ents = [ent.label_ for ent in doc_obj.ents]
    raw_ent_counts_spacy = Counter(spacy_ents)
    normalized_ent_counts_spacy = Counter()

    if total_words > 0:
        for ent_type, count in raw_ent_counts_spacy.items():
            normalized_ent_counts_spacy[ent_type] = (count / total_words) * 1000
    else:
        for ent_type, count in raw_ent_counts_spacy.items():
            normalized_ent_counts_spacy[ent_type] = 0.0

    for ent_type, count in raw_ent_counts_spacy.items():
        result_row[f'{ent_type}_spacy_raw'] = count
    for ent_type, norm_count in normalized_ent_counts_spacy.items():
        result_row[f'{ent_type}_spacy_norm'] = norm_count

    print(f"Found {len(doc_obj.ents)} entities with spaCy !")

    # ----------------------- Part 2 -----------------------
    # ------------------------------------------------------
    split_text_filename = doc_name.replace(".spacy", ".txt")
    split_text_path = os.path.join(SPLIT_TEXT_DIR, split_text_filename)

    raw_ent_counts_cc = Counter()
    normalized_ent_counts_cc = Counter()
    total_cc_ent_doc = 0

    if os.path.exists(split_text_path):
        with open(split_text_path, 'r', encoding='utf-8') as f:
            split_text = f.read()
        
        if split_text.strip():
            paragraphs = [p.strip() for p in split_text.split('\n\n') if p.strip()]
            
            if not paragraphs:
                print(f"Error: No paragraphs found")
            else:
                all_ent_classes = []
                
                for i, paragraph_text in enumerate(paragraphs):
                    
                    # CC NER processing
                    paragraph_results = cc_ner_pipeline(paragraph_text)
                    
                    for ent in paragraph_results:
                        all_ent_classes.append(ent['entity_group'])
                    total_cc_ent_doc += len(paragraph_results)

                raw_ent_counts_cc = Counter(all_ent_classes)
                print(f"CC NER complete for {len(paragraphs)} paragraphs. Found {total_cc_ent_doc} entities.")

                if total_words > 0:
                    for ent_type, count in raw_ent_counts_cc.items():
                        normalized_ent_counts_cc[ent_type] = (count / total_words) * 1000
                else:
                    for ent_type, count in raw_ent_counts_cc.items():
                        normalized_ent_counts_cc[ent_type] = 0.0
        else:
            print(f"Cannot read: '{split_text_path}'. Check content or encoding and try again")
    else:
        print(f"file not found: '{split_text_path}'. Check paths and try again")

    for ent_type, count in raw_ent_counts_cc.items():
        result_row[f'{ent_type}_cc_raw'] = count
    for ent_type, norm_count in normalized_ent_counts_cc.items():
        result_row[f'{ent_type}_cc_norm'] = norm_count
        
    all_results.append(result_row)

# Convert to DF
results_df = pd.DataFrame(all_results)
results_df = results_df.fillna(0)

fixed_cols = ['Document', 'Total Words']
    
all_ent_prefixes = set()
for col_name in results_df.columns:
    if col_name not in fixed_cols:
        parts = col_name.split('_')
        if len(parts) > 1:
            prefix_candidate = "_".join(parts[:-1])
            all_ent_prefixes.add(prefix_candidate)

sorted_ent_prefixes = sorted(list(all_ent_prefixes))
    
ent_cols = []
for prefix in sorted_ent_prefixes:
    raw_col_name = f'{prefix}_raw'
    norm_col_name = f'{prefix}_norm'
    if raw_col_name in results_df.columns:
        ent_cols.append(raw_col_name)
    if norm_col_name in results_df.columns:
        ent_cols.append(norm_col_name)
   
final_columns = fixed_cols + ent_cols
results_df = results_df[[col for col in final_columns if col in results_df.columns]]

results_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
print(f"Done. All metrics saved to: '{OUTPUT_CSV}'")
#!/usr/bin/env bash
# Fit centroids on the discovery samples and label every embedded page.
# Retain the centroids of boilerplate topics and flag their assignments.
set -euo pipefail
cd "$(dirname "$0")/.."

EMB=embeddings/dlr_embeddings_clean2_dedup.npz        # canonical, deduplicated
CORPUS=data/corpus.parquet
SUB=bertopic_min100
NONPROSE=output/cleaning/non_prose.parquet            # from detect_non_prose.py

NP_ARG=()
if [ -f "${NONPROSE}" ]; then
    NP_ARG=(--exclude_non_prose "${NONPROSE}")
    echo "using non-prose flags from ${NONPROSE}"
else
    echo "NOTE: ${NONPROSE} not found — run detect_non_prose.py first to exclude crawl artifacts"
fi

for language in eng deu; do
    echo "########## ${language} ##########"
    S2="output/stage2_clean2/${language}"

    python scripts/filter_junk_topics.py \
        --lang "${language}" --sub "${SUB}" \
        --out "output/production_labels/${language}/keywords_clean.json"

    python scripts/label_by_centroid.py \
        --lang "${language}" \
        --corpus_parquet "${CORPUS}" \
        --emb "${S2}/sample_data/sample_embeddings.npz" \
        --topics "${S2}/${SUB}/doc_topics.parquet" \
        --topic_words "${S2}/${SUB}/topic_top_words.json" \
        --exclude_junk "output/production_labels/${language}/keywords_clean.json.report.json" \
        "${NP_ARG[@]}" \
        --target_emb "${EMB}" \
        --out "output/production_labels/${language}/doc_labels.parquet"
done

echo
echo "=== done — every embedded page has a topic ==="

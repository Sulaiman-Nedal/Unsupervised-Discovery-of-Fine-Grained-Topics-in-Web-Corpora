#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
corpus="data/corpus.parquet"
embeddings="embeddings/dlr_embeddings_clean2_dedup.npz"
discovery="output/stage2_clean2"

if [[ ! -r "$corpus" ]]; then
    printf 'Expected a readable corpus at pipeline/%s\n' "$corpus" >&2
    exit 2
fi

# Step 1. Embed the cleaned pages.
python scripts/embed_pages.py \
    --input "$corpus" --output "$embeddings" \
    --min_tokens 20 --max_length 512 --chunk --max_chunks 8 \
    --batch_size "${PIPELINE_BATCH_SIZE:-16}" --num_threads "${PIPELINE_THREADS:-8}"

# Step 2. Discover topics in a sample of each language.
for language in eng deu; do
    sample="$discovery/$language/sample_data"
    python scripts/build_sample.py \
        --embeddings "$embeddings" --parquet "$corpus" --output_dir "$sample" \
        --sample_size 50000 --seed 42 --lang "$language"
    python scripts/run_bertopic.py \
        --embeddings "$sample/sample_embeddings.npz" --parquet "$sample/sample_data.parquet" \
        --output_dir "$discovery/$language/bertopic_min100" \
        --min_topic_size 100
done

# Step 3. Assign every embedded page a topic, then refine large topics.
python scripts/detect_non_prose.py --corpus "$corpus"
bash scripts/run_production_labeling.sh
for language in eng deu; do
    python scripts/apply_subdivision.py --lang "$language" --corpus "$corpus" --emb "$embeddings"
done

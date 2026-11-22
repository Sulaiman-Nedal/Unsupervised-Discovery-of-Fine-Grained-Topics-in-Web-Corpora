#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

for language in eng deu; do
    sample="../pipeline/output/stage2_clean2/$language/sample_data/sample_data.parquet"
    if [[ ! -r "$sample" ]]; then
        printf 'Missing the %s discovery sample at %s\n' "$language" "$sample" >&2
        exit 2
    fi
    PYTHONPATH="../pipeline/scripts${PYTHONPATH:+:$PYTHONPATH}" python run_lda.py \
        --parquet "$sample" --output_dir "output/$language/lda_k20" \
        --num_topics 20 --seed 42
done

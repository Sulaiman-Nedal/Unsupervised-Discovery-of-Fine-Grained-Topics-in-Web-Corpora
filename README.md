# Unsupervised Discovery of Fine-Grained Topics in Web Corpora

[Read the current thesis draft](thesis/main.pdf).

| Start here | Purpose |
|---|---|
| [Pipeline](pipeline/run.sh) | Embed the pages, discover topics with BERTopic, then label the pages. |
| [RQ3 LDA runs](rq3/run_lda.sh) | Fit the separate 20-topic LDA reference for each language. |
| [Thesis](thesis/main.tex) | LaTeX source with references and figures. |

## Set up the pipeline

Use Python 3.11 and Bash. The first embedding run downloads the pretrained
[multilingual E5 model](https://huggingface.co/intfloat/multilingual-e5-large-instruct).
Use a GPU for a large corpus. Set `PIPELINE_BATCH_SIZE` to adjust the embedding
batch size.

~~~bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
~~~

Put a Parquet file at `pipeline/data/corpus.parquet`. The corpus is supplied
separately and must have these columns.

| Column | Content |
|---|---|
| `uuid` | Unique string ID for each page. |
| `plain_text` | The page's body text. |
| `lang` | Language code. The pipeline processes `eng` and `deu`. |
| `url` | The page URL. |
| `url_path` | The path part of the URL. |
| `url_query` | The query part of the URL, or an empty string. |

The bilingual run requires enough useful text in each language for topic discovery.
For the full DLR run, use the corpus through the project's existing data access.

## Run the three pipeline steps

From the repository root, run `bash pipeline/run.sh`. The script lists all
three steps and their settings.

1. **Embed.** Clean the text and generate E5 vectors with a cap of eight chunks per
   page. The vectors are written to `pipeline/embeddings/`.
2. **Discover topics.** Draw a URL-filtered sample per language and run BERTopic
   with minimum topic size 100. The samples and assignments are written to
   `pipeline/output/stage2_clean2/`.
3. **Label.** Flag non-prose pages and boilerplate topics. Assign a topic to each
   embedded page and subdivide selected large topics. The document labels and
   topic keywords are written to `pipeline/output/production_labels/`.

To try the separate search command after labeling, run
`python pipeline/scripts/search.py --lang eng --query "satellite images" --cpu`.
The optional GPU encoder expects a complete E5 ONNX export under
`pipeline/output/onnx_e5/fp32/` and the corresponding ONNX Runtime and Optimum
packages.

## Run the LDA reference for RQ3

After the discovery step has saved the English and German samples, run
`bash rq3/run_lda.sh`. It fits LDA with 20 topics and seed 42 on each sample.
The document assignments and keywords are saved under `rq3/output/`.
These runs supply the coarse reference for the hierarchy examined in RQ3.

## Build the thesis

Open [the current PDF](thesis/main.pdf) for review. To build it from the included
source and figures with [Tectonic](https://tectonic-typesetting.github.io/), run
`tectonic -X compile main.tex` from the `thesis/` directory.

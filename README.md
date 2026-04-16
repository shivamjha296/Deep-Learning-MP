# Deep Learning Mini Project: Fake News Detection

This repository contains our end-to-end implementation for a Deep Learning mini project focused on fake news detection.

The project is structured around three academic goals:

1. Reproduce a published research baseline.
2. Identify and demonstrate a practical flaw.
3. Propose and evaluate an improved solution.

## Team Members

- Shivam Jha
- Gaurang Gade
- Sejal Chaudhari

## Project Summary

We reproduce the FakeBERT-style pipeline (BERT + BiLSTM) on the ISOT dataset, then demonstrate its weakness on short noisy social-media style text. To address this, we train an improved RoBERTa + BiLSTM model using LIAR, a dataset better aligned with short claim-like statements.

The repository provides:

- A Streamlit application for training, evaluation, and interactive testing.
- A notebook implementation for stepwise demonstration.
- Reusable training/evaluation code in a dedicated source module.
- Persistent configuration-based model caching to skip repeated retraining for identical settings.

## Research Baseline

- Base paper: FakeBERT: Fake News Detection in Social Media with a BERT-based Deep Learning Approach (2021)
- Baseline model in this project: BERT + BiLSTM
- Baseline training dataset: ISOT (clean long-form news articles)

## Identified Flaw

The baseline model is strong on clean article-style text but less robust on noisy, short, informal inputs (social forwards, slang, mixed language patterns, clickbait wording).

This flaw is explicitly demonstrated in both the Streamlit app and notebook using noisy sample inputs.

## Improvement Strategy

- Improved model: RoBERTa + BiLSTM
- Improved data domain: LIAR dataset (short political claims)
- Rationale: RoBERTa and LIAR are better aligned with short/noisy claim-style text than the original baseline setup.

## Key Implementation Features

- Two-model UI workflow with separate tabs (Baseline Model and Improved Model).
- Sidebar training controls: Train Baseline Model, Train Improved Model, and Train All Models (full pipeline).
- Custom input prediction for both models.
- Optional baseline-vs-improved prediction comparison.
- Compact centered confusion matrix display and training curve visualization.
- Configuration-aware disk cache for repeated runs.

## Configuration-Based Training Cache

The app now caches training outputs for each unique configuration.

If the same configuration is used again, the app loads cached artifacts instead of retraining.

The cache key includes:

- Model type and architecture settings.
- Epochs, learning rate, batch size, max token length, sample limits.
- Random seed.
- Dataset source and file fingerprints (path, size, modified time).

Cache location:

- `models/cache/baseline/<hash>/`
- `models/cache/improved/<hash>/`

Each cache entry stores:

- `model.pt`
- tokenizer files
- `history.csv`
- `test_result.json`
- `metadata.json`

## Repository Structure

```text
.
|-- streamlit_app.py
|-- requirements.txt
|-- README.md
|-- Dataset/
|   |-- True.csv
|   |-- Fake.csv
|   |-- train.tsv
|   |-- valid.tsv
|   \-- test.tsv
|-- notebooks/
|   \-- fake_news_detection.ipynb
|-- src/
|   |-- fake_news_core.py
|   \-- sample_noisy_inputs.py
\-- models/
    |-- fakebert_isot/
    |-- roberta_liar/
    \-- cache/
```

## Setup Instructions

### 1. Create and activate environment

Windows PowerShell:

```bash
python -m venv myenv
myenv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare datasets

ISOT:

- Place `True.csv` and `Fake.csv` in `Dataset/`
- Or provide custom file paths in the app sidebar.

LIAR (optional local mode):

- Place `train.tsv`, `valid.tsv`, `test.tsv` in `Dataset/`
- Otherwise LIAR can be loaded from Hugging Face.

## Run the Streamlit Application

```bash
streamlit run streamlit_app.py
```

Optional quieter mode:

```bash
streamlit run streamlit_app.py --server.fileWatcherType none
```

## Suggested Workflow in App

1. Set training controls in the sidebar.
2. Use one of the three training actions: Train Baseline Model, Train Improved Model, or Train All Models.
3. Review metrics, classification report, confusion matrix, and curves in each tab.
4. Test outputs using both multi-line batch testing and single custom input prediction.
5. Re-run with same config to verify cache hit and faster loading.

## Notebook Version

Notebook file:

- `notebooks/fake_news_detection.ipynb`

Use this for cell-by-cell demonstration in presentations or viva.

## Evaluation Outputs

The implementation reports:

- Accuracy, Precision, Recall, F1-score
- Classification report
- Confusion matrix
- Training curves
- Qualitative output comparison on noisy/custom text

## Notes for Academic Review

- The baseline implementation follows the selected research direction.
- The project clearly documents and demonstrates a real deployment flaw.
- The improved pipeline changes both representation and data domain.
- The system supports reproducibility through deterministic seeds and artifact caching.

## References

- FakeBERT paper (2021): https://pmc.ncbi.nlm.nih.gov/articles/PMC7788551/pdf/11042_2020_Article_10183.pdf
- ISOT dataset: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
- LIAR dataset (Hugging Face): https://huggingface.co/datasets/liar

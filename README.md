run # DL Mini Project: Fake News Detection

This project is built to match your rubric exactly:

1. Implement a research paper end-to-end.
2. Identify and demonstrate a major flaw.
3. Implement and evaluate an improved solution.

## Selected Base Paper

- FakeBERT: Fake News Detection in Social Media with a BERT-based Deep Learning Approach (2021)
- Baseline here: BERT + BiLSTM on ISOT (clean long-form English news)

## Documented Flaw Implemented

The baseline is trained on clean, balanced, long articles and does not generalize well to short, noisy social text (forwards, slang, mixed language, clickbait phrasing).

This is demonstrated explicitly using noisy WhatsApp-style samples in both notebook and app.

## Improvement Implemented

- Improved model: RoBERTa + BiLSTM
- Improved training data: LIAR dataset (short political claims, noisier than ISOT)
- Why this choice: RoBERTa generally handles noisy text better and LIAR is closer to short claim style than ISOT.

## Project Structure

- streamlit_app.py: Interactive app with all 3 phases.
- notebooks/fake_news_dl_mini_project.ipynb: Colab/Notebook implementation in ordered cells.
- src/fake_news_core.py: Shared training, evaluation, and prediction logic.
- src/sample_noisy_inputs.py: Noisy social text samples for flaw demonstration.
- models/: Saved checkpoints/artifacts after training.

## Setup (Local / VS Code)

1. Create and activate your Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

### GPU Setup (NVIDIA)

If you want training to use GPU (with CPU fallback), install CUDA-enabled PyTorch in your existing `myenv`:

```bash
myenv\Scripts\python.exe -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
```

Verify GPU is active:

```bash
myenv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Expected: `torch.cuda.is_available()` should print `True`.

3. Place ISOT files at:

- data/True.csv
- data/Fake.csv

Or update paths from app sidebar.

## Run Streamlit App

```bash
streamlit run streamlit_app.py
```

Cleaner logs (recommended):

```bash
streamlit run streamlit_app.py --server.fileWatcherType none
```

In the app:

1. Run Phase 1 (baseline reproduction).
2. Run Phase 2 (flaw demo) on noisy text.
3. Run Phase 3 (improved model).
4. Compare confusion matrices, F1, and noisy-sample behavior.

## Run Notebook (Colab or VS Code)

Notebook path:

- notebooks/fake_news_dl_mini_project.ipynb

Recommended flow:

1. Run cells in order.
2. Upload ISOT True.csv and Fake.csv when prompted (in Colab).
3. Execute all three phases.
4. Use final comparison cells and conclusion points for report/viva.

## Notes for Viva

- Baseline reproduces paper-style setup on ISOT.
- Flaw is empirically shown with noisy short social messages.
- Improvement changes both representation (RoBERTa) and data domain (LIAR).
- Final output includes quantitative metrics and qualitative robustness comparison.

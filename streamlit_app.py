import os
from datetime import datetime
from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
import torch

from src.fake_news_core import (
    LABEL_NAMES,
    SplitConfig,
    TransformerBiLSTMClassifier,
    build_dataloaders,
    evaluate_model,
    get_device,
    load_isot_dataframe,
    load_liar_binary_dataframe,
    predict_texts,
    save_experiment_artifacts,
    set_seed,
    train_model,
    train_val_test_split,
)
from src.sample_noisy_inputs import NOISY_SOCIAL_MEDIA_SAMPLES


st.set_page_config(
    page_title="DL Mini Project: Fake News",
    page_icon="NN",
    layout="wide",
)

st.markdown(
    """
    <style>
      .main {
        background: radial-gradient(circle at 20% 10%, #f4ecd8 0%, #f7f6f2 35%, #f0efe9 100%);
      }
      .hero {
        padding: 1rem 1.2rem;
        border-radius: 14px;
        background: linear-gradient(120deg, #1f3a5f, #274c77);
        color: #fff;
        margin-bottom: 1rem;
      }
      .hero h1 {
        margin: 0;
        font-size: 1.7rem;
      }
      .hero p {
        margin-top: 0.5rem;
        margin-bottom: 0;
        font-size: 0.95rem;
      }
      .phase-card {
        padding: 0.8rem;
        border-radius: 10px;
        border: 1px solid #d1d7df;
        background: #ffffff;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


if "device" not in st.session_state:
    st.session_state["device"] = get_device()

if "seed" not in st.session_state:
    st.session_state["seed"] = 42

set_seed(st.session_state["seed"])


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _create_training_logger(title: str):
    st.markdown(f"#### {title} Training Logs")
    progress = st.progress(0, text=f"{title}: preparing...")
    log_box = st.empty()
    log_lines: List[str] = []

    def log_message(message: str) -> None:
        line = f"[{_timestamp()}] {message}"
        log_lines.append(line)
        log_box.code("\n".join(log_lines[-40:]), language="text")
        print(line)

    def on_epoch(payload: Dict[str, object]) -> None:
        epoch = int(payload.get("epoch", 0))
        total = max(int(payload.get("epochs", 1)), 1)
        progress.progress(min(epoch / total, 1.0), text=f"{title}: epoch {epoch}/{total}")

        line = (
            f"Epoch {epoch}/{total} | "
            f"train_loss={float(payload.get('train_loss', 0.0)):.4f} "
            f"val_loss={float(payload.get('val_loss', 0.0)):.4f} | "
            f"train_acc={float(payload.get('train_acc', 0.0)):.4f} "
            f"val_acc={float(payload.get('val_acc', 0.0)):.4f} | "
            f"train_f1={float(payload.get('train_f1', 0.0)):.4f} "
            f"val_f1={float(payload.get('val_f1', 0.0)):.4f}"
        )
        if bool(payload.get("is_best", False)):
            line += " | best checkpoint updated"

        log_message(line)

    return log_message, on_epoch, progress


def _render_metrics(title: str, metrics: Dict[str, float]) -> None:
    st.subheader(title)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    c2.metric("Precision", f"{metrics['precision']:.4f}")
    c3.metric("Recall", f"{metrics['recall']:.4f}")
    c4.metric("F1", f"{metrics['f1']:.4f}")


def _plot_confusion_matrix(cm, title: str):
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)


def _plot_history(histories: Dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for name, hist_df in histories.items():
        if hist_df is None or hist_df.empty:
            continue
        axes[0].plot(hist_df["epoch"], hist_df["train_loss"], marker="o", label=f"{name} Train")
        axes[0].plot(hist_df["epoch"], hist_df["val_loss"], marker="o", linestyle="--", label=f"{name} Val")

        axes[1].plot(hist_df["epoch"], hist_df["train_f1"], marker="o", label=f"{name} Train")
        axes[1].plot(hist_df["epoch"], hist_df["val_f1"], marker="o", linestyle="--", label=f"{name} Val")

    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].set_title("F1 Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1")
    axes[1].legend()

    st.pyplot(fig)


def run_baseline_phase(
    true_csv_path: str,
    fake_csv_path: str,
    max_samples_per_class,
    batch_size: int,
    max_length: int,
    epochs: int,
    lr: float,
    log_message: Optional[Callable[[str], None]] = None,
    epoch_callback: Optional[Callable[[Dict[str, object]], None]] = None,
):
    if log_message:
        log_message(f"Loading ISOT data from {true_csv_path} and {fake_csv_path}")

    isot_df = load_isot_dataframe(
        true_csv_path,
        fake_csv_path,
        max_samples_per_class=max_samples_per_class,
        random_state=st.session_state["seed"],
    )

    if log_message:
        log_message(f"Loaded ISOT rows: {len(isot_df)}")

    train_df, val_df, test_df = train_val_test_split(
        isot_df,
        SplitConfig(test_size=0.2, val_size_from_train=0.1, random_state=st.session_state["seed"]),
    )

    if log_message:
        log_message(f"Split sizes -> train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        log_message("Building tokenizer and dataloaders...")

    train_loader, val_loader, test_loader, tokenizer = build_dataloaders(
        train_df,
        val_df,
        test_df,
        tokenizer_name="bert-base-uncased",
        max_length=max_length,
        batch_size=batch_size,
    )

    model = TransformerBiLSTMClassifier(
        model_name="bert-base-uncased",
        lstm_hidden_size=128,
        lstm_layers=1,
        dropout=0.3,
    )

    if log_message:
        log_message(f"Model initialized: bert-base-uncased, training for {epochs} epoch(s)")

    model, history_df = train_model(
        model,
        train_loader,
        val_loader,
        device=st.session_state["device"],
        epochs=epochs,
        lr=lr,
        weight_decay=0.01,
        epoch_callback=epoch_callback,
    )

    if log_message:
        log_message("Training finished. Running test evaluation...")

    test_result = evaluate_model(model, test_loader, st.session_state["device"])

    os.makedirs("models", exist_ok=True)
    save_experiment_artifacts(
        model=model,
        tokenizer=tokenizer,
        output_dir="models/fakebert_isot",
        model_name="bert-base-uncased",
        history_df=history_df,
        metadata={"dataset": "ISOT", "phase": "baseline_fakebert"},
    )

    if log_message:
        log_message("Saved artifacts to models/fakebert_isot")

    st.session_state["baseline_model"] = model
    st.session_state["baseline_tokenizer"] = tokenizer
    st.session_state["baseline_test_result"] = test_result
    st.session_state["baseline_history"] = history_df


def run_improved_phase(
    max_samples_per_split,
    batch_size: int,
    max_length: int,
    epochs: int,
    lr: float,
    log_message: Optional[Callable[[str], None]] = None,
    epoch_callback: Optional[Callable[[Dict[str, object]], None]] = None,
    liar_local_dir: Optional[str] = None,
    include_half_true: bool = True,
):
    if log_message:
        source_msg = f"local files at {liar_local_dir}" if liar_local_dir else "Hugging Face"
        log_message(f"Loading LIAR dataset splits from {source_msg}...")

    train_df = load_liar_binary_dataframe(
        split="train",
        max_samples=max_samples_per_split,
        random_state=st.session_state["seed"],
        local_dir=liar_local_dir,
        include_half_true=include_half_true,
    )
    val_df = load_liar_binary_dataframe(
        split="validation",
        max_samples=max(500, int(max_samples_per_split * 0.3)) if max_samples_per_split else None,
        random_state=st.session_state["seed"],
        local_dir=liar_local_dir,
        include_half_true=include_half_true,
    )
    test_df = load_liar_binary_dataframe(
        split="test",
        max_samples=max(500, int(max_samples_per_split * 0.3)) if max_samples_per_split else None,
        random_state=st.session_state["seed"],
        local_dir=liar_local_dir,
        include_half_true=include_half_true,
    )

    if log_message:
        log_message(f"LIAR split sizes -> train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        log_message("Building tokenizer and dataloaders...")

    train_loader, val_loader, test_loader, tokenizer = build_dataloaders(
        train_df,
        val_df,
        test_df,
        tokenizer_name="roberta-base",
        max_length=max_length,
        batch_size=batch_size,
    )

    model = TransformerBiLSTMClassifier(
        model_name="roberta-base",
        lstm_hidden_size=128,
        lstm_layers=1,
        dropout=0.3,
    )

    if log_message:
        log_message(f"Model initialized: roberta-base, training for {epochs} epoch(s)")

    model, history_df = train_model(
        model,
        train_loader,
        val_loader,
        device=st.session_state["device"],
        epochs=epochs,
        lr=lr,
        weight_decay=0.01,
        epoch_callback=epoch_callback,
    )

    if log_message:
        log_message("Training finished. Running test evaluation...")

    test_result = evaluate_model(model, test_loader, st.session_state["device"])

    os.makedirs("models", exist_ok=True)
    save_experiment_artifacts(
        model=model,
        tokenizer=tokenizer,
        output_dir="models/roberta_liar",
        model_name="roberta-base",
        history_df=history_df,
        metadata={"dataset": "LIAR", "phase": "improved_roberta"},
    )

    if log_message:
        log_message("Saved artifacts to models/roberta_liar")

    st.session_state["improved_model"] = model
    st.session_state["improved_tokenizer"] = tokenizer
    st.session_state["improved_test_result"] = test_result
    st.session_state["improved_history"] = history_df


st.markdown(
    """
    <div class="hero">
      <h1>DL Mini Project: FakeBERT Reproduction and Real-World Improvement</h1>
      <p>Phase 1 reproduces the original pipeline on ISOT. Phase 2 demonstrates the failure on noisy social text. Phase 3 upgrades to RoBERTa + BiLSTM on LIAR for better robustness.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("Experiment Controls")
    st.caption("Use Quick Mode for demo speed. Turn it off for closer-to-paper training.")

    quick_mode = st.checkbox("Quick Mode", value=True)

    true_default_path = "Dataset/True.csv" if os.path.exists("Dataset/True.csv") else "data/True.csv"
    fake_default_path = "Dataset/Fake.csv" if os.path.exists("Dataset/Fake.csv") else "data/Fake.csv"
    true_csv_path = st.text_input("ISOT True.csv Path", value=true_default_path)
    fake_csv_path = st.text_input("ISOT Fake.csv Path", value=fake_default_path)

    baseline_epochs = st.slider("Baseline Epochs", 1, 5, 1 if quick_mode else 3)
    improved_epochs = st.slider("Improved Epochs", 1, 5, 1 if quick_mode else 3)
    batch_size = st.select_slider("Batch Size", options=[4, 8, 16, 32], value=8 if quick_mode else 16)
    max_length = st.select_slider("Max Token Length", options=[64, 96, 128, 192, 256], value=128)

    baseline_max_samples = st.number_input(
        "ISOT Max Samples Per Class (0 = full)",
        min_value=0,
        max_value=50000,
        value=1500 if quick_mode else 0,
        step=500,
    )

    liar_max_samples = st.number_input(
        "LIAR Train Max Samples (0 = full)",
        min_value=0,
        max_value=12000,
        value=3000 if quick_mode else 0,
        step=500,
    )

    has_local_liar = all(
        os.path.exists(os.path.join("Dataset", filename))
        for filename in ("train.tsv", "valid.tsv", "test.tsv")
    )
    liar_local_default = "Dataset" if has_local_liar else ""
    liar_local_dir = st.text_input(
        "LIAR Local Directory (optional)",
        value=liar_local_default,
        help="Folder containing train.tsv, valid.tsv, and test.tsv",
    )
    include_half_true = st.checkbox("Treat half-true as REAL", value=True)

    lr = st.select_slider("Learning Rate", options=[1e-5, 2e-5, 3e-5, 5e-5], value=2e-5)

    st.markdown("---")
    run_baseline_btn = st.button("Run Phase 1: Reproduce FakeBERT")
    run_improved_btn = st.button("Run Phase 3: Train Improved Model")
    run_full_btn = st.button("Run Full Pipeline")


max_samples_baseline = None if int(baseline_max_samples) == 0 else int(baseline_max_samples)
max_samples_liar = None if int(liar_max_samples) == 0 else int(liar_max_samples)
liar_local_dir = liar_local_dir.strip() or None


if run_baseline_btn or run_full_btn:
    st.markdown("### Phase 1: Paper Reproduction (BERT + BiLSTM on ISOT)")
    if not os.path.exists(true_csv_path) or not os.path.exists(fake_csv_path):
        st.error("ISOT CSV files not found. Update the sidebar paths first.")
    else:
        baseline_log, baseline_epoch_cb, baseline_progress = _create_training_logger("Baseline")
        baseline_log(f"Using device: {st.session_state['device']}")
        with st.spinner("Training baseline model..."):
            run_baseline_phase(
                true_csv_path=true_csv_path,
                fake_csv_path=fake_csv_path,
                max_samples_per_class=max_samples_baseline,
                batch_size=batch_size,
                max_length=max_length,
                epochs=baseline_epochs,
                lr=lr,
                log_message=baseline_log,
                epoch_callback=baseline_epoch_cb,
            )
        baseline_progress.progress(1.0, text="Baseline: done")
        baseline_log("Baseline phase completed successfully.")
        st.success("Baseline complete. Artifacts saved in models/fakebert_isot")


if "baseline_test_result" in st.session_state:
    baseline_result = st.session_state["baseline_test_result"]
    _render_metrics("Baseline Test Metrics", baseline_result["metrics"])
    st.text("Classification Report")
    st.code(baseline_result["report"])
    _plot_confusion_matrix(baseline_result["confusion_matrix"], "Baseline Confusion Matrix")


if run_improved_btn or run_full_btn:
    st.markdown("### Phase 3: Improved Model (RoBERTa + BiLSTM on LIAR)")
    improved_log, improved_epoch_cb, improved_progress = _create_training_logger("Improved")
    improved_log(f"Using device: {st.session_state['device']}")
    with st.spinner("Training improved model..."):
        run_improved_phase(
            max_samples_per_split=max_samples_liar,
            batch_size=batch_size,
            max_length=max_length,
            epochs=improved_epochs,
            lr=lr,
            log_message=improved_log,
            epoch_callback=improved_epoch_cb,
            liar_local_dir=liar_local_dir,
            include_half_true=include_half_true,
        )
    improved_progress.progress(1.0, text="Improved: done")
    improved_log("Improved phase completed successfully.")
    st.success("Improved model complete. Artifacts saved in models/roberta_liar")


if "improved_test_result" in st.session_state:
    improved_result = st.session_state["improved_test_result"]
    _render_metrics("Improved Test Metrics", improved_result["metrics"])
    st.text("Classification Report")
    st.code(improved_result["report"])
    _plot_confusion_matrix(improved_result["confusion_matrix"], "Improved Confusion Matrix")


st.markdown("### Phase 2: Flaw Demonstration on Noisy Social Text")

sample_text = st.text_area(
    "Edit or add noisy WhatsApp-style lines (one per line)",
    value="\n".join(NOISY_SOCIAL_MEDIA_SAMPLES),
    height=180,
)

input_lines = [line.strip() for line in sample_text.splitlines() if line.strip()]

if st.button("Run Flaw Demo"):
    if "baseline_model" not in st.session_state or "baseline_tokenizer" not in st.session_state:
        st.error("Run Phase 1 first to get baseline predictions.")
    else:
        baseline_preds = predict_texts(
            model=st.session_state["baseline_model"],
            tokenizer=st.session_state["baseline_tokenizer"],
            texts=input_lines,
            device=st.session_state["device"],
            max_length=max_length,
        )

        improved_available = "improved_model" in st.session_state and "improved_tokenizer" in st.session_state
        improved_preds: List[Dict[str, object]] = []
        if improved_available:
            improved_preds = predict_texts(
                model=st.session_state["improved_model"],
                tokenizer=st.session_state["improved_tokenizer"],
                texts=input_lines,
                device=st.session_state["device"],
                max_length=max_length,
            )

        rows = []
        for idx, base in enumerate(baseline_preds):
            row = {
                "text": base["text"],
                "expected": "FAKE",
                "baseline_pred": base["label"],
                "baseline_conf": round(base["confidence"], 4),
                "baseline_correct": base["label"] == "FAKE",
            }

            if improved_available:
                improved = improved_preds[idx]
                row["improved_pred"] = improved["label"]
                row["improved_conf"] = round(improved["confidence"], 4)
                row["improved_correct"] = improved["label"] == "FAKE"
                row["changed"] = "Yes" if improved["label"] != base["label"] else "No"

            rows.append(row)

        result_df = pd.DataFrame(rows)
        st.dataframe(result_df, use_container_width=True)

        baseline_wrong = (~result_df["baseline_correct"]).sum()
        st.warning(
            f"Baseline wrong on {baseline_wrong}/{len(result_df)} noisy samples. "
            "This demonstrates the paper flaw: trained on clean, long-form English articles, not social media noise."
        )

        if improved_available:
            improved_wrong = (~result_df["improved_correct"]).sum()
            st.success(
                f"Improved model wrong on {improved_wrong}/{len(result_df)} noisy samples. "
                "Lower is better."
            )


if "baseline_history" in st.session_state or "improved_history" in st.session_state:
    st.markdown("### Training Curves")
    _plot_history(
        {
            "Baseline": st.session_state.get("baseline_history"),
            "Improved": st.session_state.get("improved_history"),
        }
    )


st.markdown("### Live Single-Text Prediction")
user_input = st.text_input("Type any claim/article text")

if st.button("Predict"):
    if not user_input.strip():
        st.error("Please enter some text.")
    elif "improved_model" in st.session_state and "improved_tokenizer" in st.session_state:
        pred = predict_texts(
            model=st.session_state["improved_model"],
            tokenizer=st.session_state["improved_tokenizer"],
            texts=[user_input],
            device=st.session_state["device"],
            max_length=max_length,
        )[0]
        st.info(f"Prediction: {pred['label']} (confidence: {pred['confidence']:.4f})")
    elif "baseline_model" in st.session_state and "baseline_tokenizer" in st.session_state:
        pred = predict_texts(
            model=st.session_state["baseline_model"],
            tokenizer=st.session_state["baseline_tokenizer"],
            texts=[user_input],
            device=st.session_state["device"],
            max_length=max_length,
        )[0]
        st.info(
            "Improved model not trained yet. "
            f"Baseline prediction: {pred['label']} (confidence: {pred['confidence']:.4f})"
        )
    else:
        st.error("Train at least one model first.")


st.caption(
    "Project framing: FakeBERT (2021) reproduction, documented flaw on noisy social media text, "
    "and improved robustness via RoBERTa + BiLSTM trained on short/noisy claims."
)

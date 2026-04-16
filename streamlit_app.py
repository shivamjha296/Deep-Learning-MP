import os
import hashlib
import json
from datetime import datetime
from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
import torch
from transformers import AutoTokenizer

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
            section[data-testid="stSidebar"] div[data-baseweb="input"] input {
                min-height: 2.1rem;
                padding-top: 0.2rem;
                padding-bottom: 0.2rem;
            }
            section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
                min-height: 2.1rem;
            }
            section[data-testid="stSidebar"] .stNumberInput input {
                min-height: 2.1rem;
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


CACHE_SCHEMA_VERSION = 1
CACHE_ROOT_DIR = os.path.join("models", "cache")


def _file_fingerprint(path: str) -> Dict[str, object]:
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        return {
            "path": abs_path,
            "exists": False,
        }

    stat = os.stat(abs_path)
    return {
        "path": abs_path,
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _build_baseline_cache_config(
    true_csv_path: str,
    fake_csv_path: str,
    max_samples_per_class,
    batch_size: int,
    max_length: int,
    epochs: int,
    lr: float,
) -> Dict[str, object]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "phase": "baseline",
        "model_name": "bert-base-uncased",
        "seed": int(st.session_state["seed"]),
        "epochs": int(epochs),
        "lr": float(lr),
        "batch_size": int(batch_size),
        "max_length": int(max_length),
        "max_samples_per_class": None if max_samples_per_class is None else int(max_samples_per_class),
        "dataset": {
            "true_csv": _file_fingerprint(true_csv_path),
            "fake_csv": _file_fingerprint(fake_csv_path),
        },
        "architecture": {
            "lstm_hidden_size": 128,
            "lstm_layers": 1,
            "dropout": 0.3,
        },
    }


def _build_improved_cache_config(
    max_samples_per_split,
    batch_size: int,
    max_length: int,
    epochs: int,
    lr: float,
    liar_local_dir: Optional[str],
    include_half_true: bool,
) -> Dict[str, object]:
    local_signatures: Dict[str, object] = {}
    local_dir_abs: Optional[str] = None
    if liar_local_dir:
        local_dir_abs = os.path.abspath(liar_local_dir)
        for filename in ("train.tsv", "valid.tsv", "test.tsv"):
            local_signatures[filename] = _file_fingerprint(os.path.join(local_dir_abs, filename))

    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "phase": "improved",
        "model_name": "roberta-base",
        "seed": int(st.session_state["seed"]),
        "epochs": int(epochs),
        "lr": float(lr),
        "batch_size": int(batch_size),
        "max_length": int(max_length),
        "max_samples_per_split": None if max_samples_per_split is None else int(max_samples_per_split),
        "include_half_true": bool(include_half_true),
        "liar_source": "local" if liar_local_dir else "huggingface",
        "liar_local_dir": local_dir_abs,
        "liar_local_files": local_signatures,
        "architecture": {
            "lstm_hidden_size": 128,
            "lstm_layers": 1,
            "dropout": 0.3,
        },
    }


def _cache_location(phase: str, config: Dict[str, object]) -> Dict[str, str]:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    cache_key = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    cache_dir = os.path.join(CACHE_ROOT_DIR, phase, cache_key)
    return {
        "key": cache_key,
        "dir": cache_dir,
    }


def _serialize_test_result(test_result: Dict[str, object]) -> Dict[str, object]:
    payload = dict(test_result)
    confusion_matrix = payload.get("confusion_matrix")
    if hasattr(confusion_matrix, "tolist"):
        payload["confusion_matrix"] = confusion_matrix.tolist()
    return payload


def _load_cached_artifacts(
    phase: str,
    config: Dict[str, object],
    model_name: str,
    log_message: Optional[Callable[[str], None]] = None,
) -> Optional[Dict[str, object]]:
    location = _cache_location(phase=phase, config=config)
    cache_dir = location["dir"]

    model_path = os.path.join(cache_dir, "model.pt")
    tokenizer_config_path = os.path.join(cache_dir, "tokenizer_config.json")
    history_path = os.path.join(cache_dir, "history.csv")
    test_result_path = os.path.join(cache_dir, "test_result.json")

    required_files = [
        model_path,
        tokenizer_config_path,
        history_path,
        test_result_path,
    ]

    if not all(os.path.exists(file_path) for file_path in required_files):
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(cache_dir)
        model = TransformerBiLSTMClassifier(
            model_name=model_name,
            lstm_hidden_size=128,
            lstm_layers=1,
            dropout=0.3,
        )
        state_dict = torch.load(model_path, map_location=st.session_state["device"])
        model.load_state_dict(state_dict)
        model = model.to(st.session_state["device"])
        model.eval()

        history_df = pd.read_csv(history_path)
        with open(test_result_path, "r", encoding="utf-8") as file_obj:
            test_result = json.load(file_obj)

        if log_message:
            log_message(f"Cache hit ({location['key']}): loaded trained {phase} model from {cache_dir}")

        return {
            "model": model,
            "tokenizer": tokenizer,
            "history_df": history_df,
            "test_result": test_result,
            "cache_dir": cache_dir,
            "cache_key": location["key"],
        }
    except Exception as exc:
        if log_message:
            log_message(f"Cache load failed, retraining {phase} model. Reason: {exc}")
        return None


def _save_cached_artifacts(
    phase: str,
    config: Dict[str, object],
    model: torch.nn.Module,
    tokenizer,
    history_df: pd.DataFrame,
    test_result: Dict[str, object],
    model_name: str,
    log_message: Optional[Callable[[str], None]] = None,
) -> None:
    location = _cache_location(phase=phase, config=config)
    cache_dir = location["dir"]
    os.makedirs(cache_dir, exist_ok=True)

    save_experiment_artifacts(
        model=model,
        tokenizer=tokenizer,
        output_dir=cache_dir,
        model_name=model_name,
        history_df=history_df,
        metadata={
            "phase": phase,
            "cache_key": location["key"],
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "config": config,
        },
    )

    with open(os.path.join(cache_dir, "test_result.json"), "w", encoding="utf-8") as file_obj:
        json.dump(_serialize_test_result(test_result), file_obj, indent=2)

    if log_message:
        log_message(f"Cached {phase} model run as key {location['key']}")


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
    fig, ax = plt.subplots(figsize=(2.8, 2.3), dpi=140)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        square=True,
        annot_kws={"size": 8},
        cbar=False,
        ax=ax,
    )
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("Actual", fontsize=8)
    ax.tick_params(axis="both", labelsize=8)
    plt.tight_layout(pad=0.5)

    _, center_col, _ = st.columns([1, 1.2, 1])
    with center_col:
        st.pyplot(fig, use_container_width=False)

    plt.close(fig)


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
    cache_config = _build_baseline_cache_config(
        true_csv_path=true_csv_path,
        fake_csv_path=fake_csv_path,
        max_samples_per_class=max_samples_per_class,
        batch_size=batch_size,
        max_length=max_length,
        epochs=epochs,
        lr=lr,
    )
    cached = _load_cached_artifacts(
        phase="baseline",
        config=cache_config,
        model_name="bert-base-uncased",
        log_message=log_message,
    )
    if cached is not None:
        st.session_state["baseline_model"] = cached["model"]
        st.session_state["baseline_tokenizer"] = cached["tokenizer"]
        st.session_state["baseline_test_result"] = cached["test_result"]
        st.session_state["baseline_history"] = cached["history_df"]
        return

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

    _save_cached_artifacts(
        phase="baseline",
        config=cache_config,
        model=model,
        tokenizer=tokenizer,
        history_df=history_df,
        test_result=test_result,
        model_name="bert-base-uncased",
        log_message=log_message,
    )

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
    cache_config = _build_improved_cache_config(
        max_samples_per_split=max_samples_per_split,
        batch_size=batch_size,
        max_length=max_length,
        epochs=epochs,
        lr=lr,
        liar_local_dir=liar_local_dir,
        include_half_true=include_half_true,
    )
    cached = _load_cached_artifacts(
        phase="improved",
        config=cache_config,
        model_name="roberta-base",
        log_message=log_message,
    )
    if cached is not None:
        st.session_state["improved_model"] = cached["model"]
        st.session_state["improved_tokenizer"] = cached["tokenizer"]
        st.session_state["improved_test_result"] = cached["test_result"]
        st.session_state["improved_history"] = cached["history_df"]
        return

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

    _save_cached_artifacts(
        phase="improved",
        config=cache_config,
        model=model,
        tokenizer=tokenizer,
        history_df=history_df,
        test_result=test_result,
        model_name="roberta-base",
        log_message=log_message,
    )

    st.session_state["improved_model"] = model
    st.session_state["improved_tokenizer"] = tokenizer
    st.session_state["improved_test_result"] = test_result
    st.session_state["improved_history"] = history_df


def _train_baseline_ui(
    true_csv_path: str,
    fake_csv_path: str,
    max_samples_per_class,
    batch_size: int,
    max_length: int,
    epochs: int,
    lr: float,
) -> bool:
    if not os.path.exists(true_csv_path) or not os.path.exists(fake_csv_path):
        st.error("ISOT CSV files not found. Update the sidebar paths first.")
        return False

    baseline_log, baseline_epoch_cb, baseline_progress = _create_training_logger("Baseline")
    baseline_log(f"Using device: {st.session_state['device']}")
    with st.spinner("Training baseline model..."):
        run_baseline_phase(
            true_csv_path=true_csv_path,
            fake_csv_path=fake_csv_path,
            max_samples_per_class=max_samples_per_class,
            batch_size=batch_size,
            max_length=max_length,
            epochs=epochs,
            lr=lr,
            log_message=baseline_log,
            epoch_callback=baseline_epoch_cb,
        )
    baseline_progress.progress(1.0, text="Baseline: done")
    baseline_log("Baseline phase completed successfully.")
    st.success("Baseline complete. Artifacts saved in models/fakebert_isot")
    return True


def _train_improved_ui(
    max_samples_per_split,
    batch_size: int,
    max_length: int,
    epochs: int,
    lr: float,
    liar_local_dir: Optional[str],
    include_half_true: bool,
) -> bool:
    improved_log, improved_epoch_cb, improved_progress = _create_training_logger("Improved")
    improved_log(f"Using device: {st.session_state['device']}")
    with st.spinner("Training improved model..."):
        run_improved_phase(
            max_samples_per_split=max_samples_per_split,
            batch_size=batch_size,
            max_length=max_length,
            epochs=epochs,
            lr=lr,
            log_message=improved_log,
            epoch_callback=improved_epoch_cb,
            liar_local_dir=liar_local_dir,
            include_half_true=include_half_true,
        )
    improved_progress.progress(1.0, text="Improved: done")
    improved_log("Improved phase completed successfully.")
    st.success("Improved model complete. Artifacts saved in models/roberta_liar")
    return True


st.markdown(
    """
    <div class="hero">
            <h1>Fake News Detection Dashboard</h1>
            <p>Train and test both models from separate tabs with clean, side-by-side style outputs.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("Experiment Controls")
    st.caption("Tune settings once, then train from each tab.")

    true_default_path = "Dataset/True.csv" if os.path.exists("Dataset/True.csv") else "data/True.csv"
    fake_default_path = "Dataset/Fake.csv" if os.path.exists("Dataset/Fake.csv") else "data/Fake.csv"

    quick_mode = st.checkbox("Quick Mode", value=True)

    baseline_epochs = st.slider("Baseline Epochs", 1, 5, 1 if quick_mode else 3)
    improved_epochs = st.slider("Improved Epochs", 1, 5, 1 if quick_mode else 3)
    batch_size = st.select_slider("Batch Size", options=[4, 8, 16, 32], value=8 if quick_mode else 16)
    max_length = st.select_slider("Max Token Length", options=[64, 96, 128, 192, 256], value=128)

    st.markdown("##### ISOT Dataset Paths")
    true_csv_path = st.text_input("True.csv Path", value=true_default_path)
    fake_csv_path = st.text_input("Fake.csv Path", value=fake_default_path)

    lr = st.select_slider("Learning Rate", options=[1e-5, 2e-5, 3e-5, 5e-5], value=2e-5)

    with st.expander("Advanced Options", expanded=False):
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

    # Defaults when advanced options remain collapsed and untouched.
    if "baseline_max_samples" not in locals():
        baseline_max_samples = 1500 if quick_mode else 0
    if "liar_max_samples" not in locals():
        liar_max_samples = 3000 if quick_mode else 0
    if "liar_local_dir" not in locals():
        liar_local_dir = "Dataset" if all(
            os.path.exists(os.path.join("Dataset", filename))
            for filename in ("train.tsv", "valid.tsv", "test.tsv")
        ) else ""
    if "include_half_true" not in locals():
        include_half_true = True

    st.markdown("---")
    st.markdown("##### Training")
    run_baseline_sidebar_btn = st.button(
        "Train Baseline Model",
        key="run_baseline_sidebar_btn",
        use_container_width=True,
    )
    run_improved_sidebar_btn = st.button(
        "Train Improved Model",
        key="run_improved_sidebar_btn",
        use_container_width=True,
    )
    run_all_sidebar_btn = st.button(
        "Train All Models",
        key="run_all_sidebar_btn",
        type="primary",
        use_container_width=True,
    )
    st.caption("Tab buttons are still available for per-model training.")


max_samples_baseline = None if int(baseline_max_samples) == 0 else int(baseline_max_samples)
max_samples_liar = None if int(liar_max_samples) == 0 else int(liar_max_samples)
liar_local_dir = liar_local_dir.strip() or None
sample_seed_text = "\n".join(NOISY_SOCIAL_MEDIA_SAMPLES[:2])
train_baseline_from_sidebar = run_baseline_sidebar_btn
train_improved_from_sidebar = run_improved_sidebar_btn


if run_all_sidebar_btn:
    st.markdown("### Full Pipeline Training")
    _train_baseline_ui(
        true_csv_path=true_csv_path,
        fake_csv_path=fake_csv_path,
        max_samples_per_class=max_samples_baseline,
        batch_size=batch_size,
        max_length=max_length,
        epochs=baseline_epochs,
        lr=lr,
    )
    _train_improved_ui(
        max_samples_per_split=max_samples_liar,
        batch_size=batch_size,
        max_length=max_length,
        epochs=improved_epochs,
        lr=lr,
        liar_local_dir=liar_local_dir,
        include_half_true=include_half_true,
    )


baseline_tab, improved_tab = st.tabs(
    [
        "Baseline Model",
        "Improved Model",
    ]
)


with baseline_tab:
    st.markdown("### Baseline: FakeBERT on ISOT")
    st.caption("Train and test the baseline model here.")

    status_col, action_col = st.columns([2, 1])
    with status_col:
        if "baseline_test_result" in st.session_state:
            st.success("Baseline model is trained and ready for testing.")
        else:
            st.info("Baseline model is not trained yet.")
    with action_col:
        train_baseline_btn = st.button(
            "Train Baseline",
            key="train_baseline_btn",
            type="primary",
            use_container_width=True,
        )

    if train_baseline_btn or train_baseline_from_sidebar:
        _train_baseline_ui(
            true_csv_path=true_csv_path,
            fake_csv_path=fake_csv_path,
            max_samples_per_class=max_samples_baseline,
            batch_size=batch_size,
            max_length=max_length,
            epochs=baseline_epochs,
            lr=lr,
        )

    if "baseline_test_result" in st.session_state:
        baseline_result = st.session_state["baseline_test_result"]
        _render_metrics("Baseline Test Metrics", baseline_result["metrics"])
        with st.expander("Show Classification Report", expanded=False):
            st.code(baseline_result["report"])
        _plot_confusion_matrix(baseline_result["confusion_matrix"], "Baseline Confusion Matrix")

        if "baseline_history" in st.session_state:
            with st.expander("Show Baseline Training Curves", expanded=False):
                _plot_history({"Baseline": st.session_state.get("baseline_history")})

    st.markdown("#### Baseline Output Test")
    st.caption("Two short example lines are loaded by default.")
    baseline_input_text = st.text_area(
        "Test text lines (one per line)",
        value=sample_seed_text,
        height=120,
        key="baseline_input_text",
    )

    if st.button("Test Baseline Outputs", key="test_baseline_outputs_btn", use_container_width=True):
        input_lines = [line.strip() for line in baseline_input_text.splitlines() if line.strip()]

        if not input_lines:
            st.error("Please enter at least one non-empty line.")
        elif "baseline_model" not in st.session_state or "baseline_tokenizer" not in st.session_state:
            st.error("Train the baseline model first.")
        else:
            baseline_preds = predict_texts(
                model=st.session_state["baseline_model"],
                tokenizer=st.session_state["baseline_tokenizer"],
                texts=input_lines,
                device=st.session_state["device"],
                max_length=max_length,
            )

            baseline_rows = [
                {
                    "text": pred["text"],
                    "prediction": pred["label"],
                    "confidence": round(pred["confidence"], 4),
                }
                for pred in baseline_preds
            ]
            baseline_df = pd.DataFrame(baseline_rows)
            st.dataframe(
                baseline_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "text": st.column_config.TextColumn("Input Text", width="large"),
                    "prediction": st.column_config.TextColumn("Prediction", width="small"),
                    "confidence": st.column_config.NumberColumn("Confidence", format="%.4f"),
                },
            )

    st.markdown("#### Baseline Custom Input")
    baseline_custom_text = st.text_area(
        "Enter custom text",
        value="",
        height=90,
        key="baseline_custom_text",
        placeholder="Type any claim/news text to test baseline model...",
    )

    if st.button("Predict Baseline Custom Input", key="predict_baseline_custom_btn", use_container_width=True):
        if not baseline_custom_text.strip():
            st.error("Please enter custom text for prediction.")
        elif "baseline_model" not in st.session_state or "baseline_tokenizer" not in st.session_state:
            st.error("Train the baseline model first.")
        else:
            baseline_custom_pred = predict_texts(
                model=st.session_state["baseline_model"],
                tokenizer=st.session_state["baseline_tokenizer"],
                texts=[baseline_custom_text.strip()],
                device=st.session_state["device"],
                max_length=max_length,
            )[0]

            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "model": "Baseline",
                            "prediction": baseline_custom_pred["label"],
                            "confidence": round(baseline_custom_pred["confidence"], 4),
                        }
                    ]
                ),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "model": st.column_config.TextColumn("Model", width="small"),
                    "prediction": st.column_config.TextColumn("Prediction", width="small"),
                    "confidence": st.column_config.NumberColumn("Confidence", format="%.4f"),
                },
            )


with improved_tab:
    st.markdown("### Improved: RoBERTa + BiLSTM")
    st.caption("Train and test the improved model here.")

    status_col, action_col = st.columns([2, 1])
    with status_col:
        if "improved_test_result" in st.session_state:
            st.success("Improved model is trained and ready for testing.")
        else:
            st.info("Improved model is not trained yet.")
    with action_col:
        train_improved_btn = st.button(
            "Train Improved",
            key="train_improved_btn",
            type="primary",
            use_container_width=True,
        )

    if train_improved_btn or train_improved_from_sidebar:
        _train_improved_ui(
            max_samples_per_split=max_samples_liar,
            batch_size=batch_size,
            max_length=max_length,
            epochs=improved_epochs,
            lr=lr,
            liar_local_dir=liar_local_dir,
            include_half_true=include_half_true,
        )

    if "improved_test_result" in st.session_state:
        improved_result = st.session_state["improved_test_result"]
        _render_metrics("Improved Test Metrics", improved_result["metrics"])
        with st.expander("Show Classification Report", expanded=False):
            st.code(improved_result["report"])
        _plot_confusion_matrix(improved_result["confusion_matrix"], "Improved Confusion Matrix")

        if "improved_history" in st.session_state:
            with st.expander("Show Improved Training Curves", expanded=False):
                _plot_history({"Improved": st.session_state.get("improved_history")})

    st.markdown("#### Improved Output Test")
    st.caption("Two short example lines are loaded by default.")
    improved_input_text = st.text_area(
        "Test text lines (one per line)",
        value=sample_seed_text,
        height=120,
        key="improved_input_text",
    )
    compare_with_baseline = st.checkbox(
        "Compare with baseline outputs when baseline is available",
        value=True,
        key="compare_with_baseline",
    )

    if st.button("Test Improved Outputs", key="test_improved_outputs_btn", use_container_width=True):
        input_lines = [line.strip() for line in improved_input_text.splitlines() if line.strip()]

        if not input_lines:
            st.error("Please enter at least one non-empty line.")
        elif "improved_model" not in st.session_state or "improved_tokenizer" not in st.session_state:
            st.error("Train the improved model first.")
        else:
            improved_preds = predict_texts(
                model=st.session_state["improved_model"],
                tokenizer=st.session_state["improved_tokenizer"],
                texts=input_lines,
                device=st.session_state["device"],
                max_length=max_length,
            )

            baseline_available = (
                compare_with_baseline
                and "baseline_model" in st.session_state
                and "baseline_tokenizer" in st.session_state
            )

            baseline_preds: List[Dict[str, object]] = []
            if baseline_available:
                baseline_preds = predict_texts(
                    model=st.session_state["baseline_model"],
                    tokenizer=st.session_state["baseline_tokenizer"],
                    texts=input_lines,
                    device=st.session_state["device"],
                    max_length=max_length,
                )

            improved_rows = []
            for idx, improved_pred in enumerate(improved_preds):
                row = {
                    "text": improved_pred["text"],
                    "improved_pred": improved_pred["label"],
                    "improved_conf": round(improved_pred["confidence"], 4),
                }

                if baseline_available:
                    baseline_pred = baseline_preds[idx]
                    row["baseline_pred"] = baseline_pred["label"]
                    row["baseline_conf"] = round(baseline_pred["confidence"], 4)
                    row["changed"] = "Yes" if baseline_pred["label"] != improved_pred["label"] else "No"

                improved_rows.append(row)

            improved_df = pd.DataFrame(improved_rows)
            improved_columns = {
                "text": st.column_config.TextColumn("Input Text", width="large"),
                "improved_pred": st.column_config.TextColumn("Improved", width="small"),
                "improved_conf": st.column_config.NumberColumn("Improved Conf", format="%.4f"),
            }
            if baseline_available:
                improved_columns["baseline_pred"] = st.column_config.TextColumn("Baseline", width="small")
                improved_columns["baseline_conf"] = st.column_config.NumberColumn("Baseline Conf", format="%.4f")
                improved_columns["changed"] = st.column_config.TextColumn("Changed", width="small")

            st.dataframe(
                improved_df,
                use_container_width=True,
                hide_index=True,
                column_config=improved_columns,
            )

            if baseline_available:
                changed_count = (improved_df["changed"] == "Yes").sum()
                st.info(f"Predictions changed on {changed_count}/{len(improved_df)} samples versus baseline.")
            elif compare_with_baseline:
                st.caption("Baseline model is not trained yet, so comparison is skipped.")

    st.markdown("#### Improved Custom Input")
    improved_custom_text = st.text_area(
        "Enter custom text",
        value="",
        height=90,
        key="improved_custom_text",
        placeholder="Type any claim/news text to test improved model...",
    )
    compare_custom_with_baseline = st.checkbox(
        "Also compare baseline for this custom text",
        value=True,
        key="compare_custom_with_baseline",
    )

    if st.button("Predict Improved Custom Input", key="predict_improved_custom_btn", use_container_width=True):
        if not improved_custom_text.strip():
            st.error("Please enter custom text for prediction.")
        elif "improved_model" not in st.session_state or "improved_tokenizer" not in st.session_state:
            st.error("Train the improved model first.")
        else:
            improved_custom_pred = predict_texts(
                model=st.session_state["improved_model"],
                tokenizer=st.session_state["improved_tokenizer"],
                texts=[improved_custom_text.strip()],
                device=st.session_state["device"],
                max_length=max_length,
            )[0]

            custom_rows = [
                {
                    "model": "Improved",
                    "prediction": improved_custom_pred["label"],
                    "confidence": round(improved_custom_pred["confidence"], 4),
                }
            ]

            baseline_custom_available = (
                compare_custom_with_baseline
                and "baseline_model" in st.session_state
                and "baseline_tokenizer" in st.session_state
            )
            if baseline_custom_available:
                baseline_custom_pred = predict_texts(
                    model=st.session_state["baseline_model"],
                    tokenizer=st.session_state["baseline_tokenizer"],
                    texts=[improved_custom_text.strip()],
                    device=st.session_state["device"],
                    max_length=max_length,
                )[0]
                custom_rows.append(
                    {
                        "model": "Baseline",
                        "prediction": baseline_custom_pred["label"],
                        "confidence": round(baseline_custom_pred["confidence"], 4),
                    }
                )
            elif compare_custom_with_baseline:
                st.caption("Baseline model is not trained yet, so custom comparison is skipped.")

            st.dataframe(
                pd.DataFrame(custom_rows),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "model": st.column_config.TextColumn("Model", width="small"),
                    "prediction": st.column_config.TextColumn("Prediction", width="small"),
                    "confidence": st.column_config.NumberColumn("Confidence", format="%.4f"),
                },
            )


if "baseline_history" in st.session_state and "improved_history" in st.session_state:
    st.markdown("### Combined Training Curves")
    with st.expander("Show Baseline vs Improved Curves", expanded=False):
        _plot_history(
            {
                "Baseline": st.session_state.get("baseline_history"),
                "Improved": st.session_state.get("improved_history"),
            }
        )


st.caption(
    "Tip: Use short noisy lines in each tab to compare model behavior quickly."
)

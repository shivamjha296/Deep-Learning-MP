import json
import os
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


LABEL_NAMES = ["FAKE", "REAL"]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class SplitConfig:
    test_size: float = 0.2
    val_size_from_train: float = 0.1
    random_state: int = 42


class EncodedNewsDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        tokenizer,
        max_length: int = 256,
    ) -> None:
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }


class TransformerBiLSTMClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        lstm_hidden_size: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.3,
        freeze_transformer: bool = False,
        num_labels: int = 2,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.transformer = AutoModel.from_pretrained(model_name)

        if freeze_transformer:
            for param in self.transformer.parameters():
                param.requires_grad = False

        transformer_hidden = self.transformer.config.hidden_size
        self.lstm = nn.LSTM(
            input_size=transformer_hidden,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_labels),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        transformer_output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = transformer_output.last_hidden_state
        lstm_output, _ = self.lstm(sequence_output)

        mask = attention_mask.unsqueeze(-1)
        masked_lstm = lstm_output * mask
        summed = masked_lstm.sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        pooled = summed / lengths

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


def _clean_text(value: str) -> str:
    text = "" if pd.isna(value) else str(value)
    return " ".join(text.split())


def load_isot_dataframe(
    true_csv_path: str,
    fake_csv_path: str,
    max_samples_per_class: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    true_df = pd.read_csv(true_csv_path)
    fake_df = pd.read_csv(fake_csv_path)

    true_df["label"] = 1
    fake_df["label"] = 0

    if max_samples_per_class is not None:
        true_df = true_df.sample(n=min(max_samples_per_class, len(true_df)), random_state=random_state)
        fake_df = fake_df.sample(n=min(max_samples_per_class, len(fake_df)), random_state=random_state)

    combined = pd.concat([true_df, fake_df], ignore_index=True)
    title = combined.get("title", "")
    body = combined.get("text", "")
    combined["text"] = (title.fillna("") + " . " + body.fillna("")).map(_clean_text)
    combined = combined[["text", "label"]].dropna().reset_index(drop=True)
    return combined.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def _liar_to_binary(label_id: int) -> int:
    # LIAR labels: 0 pants-fire, 1 false, 2 barely-true, 3 half-true, 4 mostly-true, 5 true
    return 0 if int(label_id) <= 2 else 1


LIAR_COLUMNS = [
    "id",
    "label",
    "statement",
    "subject",
    "speaker",
    "speaker_job",
    "state",
    "party",
    "barely_true",
    "false",
    "half_true",
    "mostly_true",
    "pants_fire",
    "context",
]

LIAR_SPLIT_FILES = {
    "train": "train.tsv",
    "validation": "valid.tsv",
    "valid": "valid.tsv",
    "val": "valid.tsv",
    "test": "test.tsv",
}


def _liar_label_to_binary(label_value: object, include_half_true: bool = True) -> Optional[int]:
    if pd.isna(label_value):
        return None

    if isinstance(label_value, (int, np.integer)):
        return _liar_to_binary(int(label_value))

    label_str = str(label_value).strip().lower().replace("_", "-")
    if label_str.isdigit():
        return _liar_to_binary(int(label_str))

    fake_labels = {"false", "pants-fire", "barely-true"}
    real_labels = {"mostly-true", "true"}
    if include_half_true:
        real_labels.add("half-true")

    if label_str in fake_labels:
        return 0
    if label_str in real_labels:
        return 1

    return None


def _load_liar_split_from_local(
    split: str,
    local_dir: str,
    include_half_true: bool = True,
) -> pd.DataFrame:
    split_key = split.strip().lower()
    filename = LIAR_SPLIT_FILES.get(split_key)
    if filename is None:
        raise ValueError(f"Unsupported LIAR split '{split}'. Use train, validation, or test.")

    file_path = os.path.join(local_dir, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"LIAR split file not found: {file_path}")

    raw_df = pd.read_csv(file_path, sep="\t", header=None, names=LIAR_COLUMNS)
    raw_df["text"] = raw_df["statement"].map(_clean_text)
    raw_df["label"] = raw_df["label"].map(
        lambda x: _liar_label_to_binary(x, include_half_true=include_half_true)
    )

    df = raw_df[["text", "label"]].dropna()
    df["label"] = df["label"].astype(int)
    return df


def load_liar_binary_dataframe(
    split: str = "train",
    max_samples: Optional[int] = None,
    random_state: int = 42,
    local_dir: Optional[str] = None,
    include_half_true: bool = True,
) -> pd.DataFrame:
    if local_dir:
        df = _load_liar_split_from_local(
            split=split,
            local_dir=local_dir,
            include_half_true=include_half_true,
        )
    else:
        dataset_split = load_dataset("liar", split=split)
        df = dataset_split.to_pandas()
        df["text"] = df["statement"].map(_clean_text)
        df["label"] = df["label"].map(_liar_to_binary)
        df = df[["text", "label"]].dropna()
        df["label"] = df["label"].astype(int)

    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=random_state)

    return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def train_val_test_split(
    df: pd.DataFrame,
    split_cfg: SplitConfig = SplitConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=split_cfg.test_size,
        random_state=split_cfg.random_state,
        stratify=df["label"],
    )

    train_df, val_df = train_test_split(
        train_df,
        test_size=split_cfg.val_size_from_train,
        random_state=split_cfg.random_state,
        stratify=train_df["label"],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer_name: str,
    max_length: int = 256,
    batch_size: int = 16,
) -> Tuple[DataLoader, DataLoader, DataLoader, object]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    train_ds = EncodedNewsDataset(train_df["text"], train_df["label"], tokenizer, max_length=max_length)
    val_ds = EncodedNewsDataset(val_df["text"], val_df["label"], tokenizer, max_length=max_length)
    test_ds = EncodedNewsDataset(test_df["text"], test_df["label"], tokenizer, max_length=max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, tokenizer


def _classification_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    device: torch.device,
    train_mode: bool,
    max_grad_norm: float = 1.0,
) -> Tuple[float, List[int], List[int], List[List[float]]]:
    criterion = nn.CrossEntropyLoss()
    epoch_loss = 0.0
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[List[float]] = []

    model.train(mode=train_mode)

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if train_mode:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train_mode):
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)

            if train_mode:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        epoch_loss += float(loss.item())
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
        y_prob.extend(probs.detach().cpu().tolist())

    avg_loss = epoch_loss / max(len(dataloader), 1)
    return avg_loss, y_true, y_pred, y_prob


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 2,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    epoch_callback: Optional[Callable[[Dict[str, object]], None]] = None,
) -> Tuple[nn.Module, pd.DataFrame]:
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = epochs * max(1, len(train_loader))
    warmup_steps = max(1, int(0.1 * total_steps))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_f1 = -1.0
    best_state = None
    history_rows = []

    for epoch in range(1, epochs + 1):
        train_loss, train_true, train_pred, _ = run_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            train_mode=True,
        )
        val_loss, val_true, val_pred, _ = run_epoch(
            model,
            val_loader,
            optimizer=None,
            scheduler=None,
            device=device,
            train_mode=False,
        )

        train_metrics = _classification_metrics(train_true, train_pred)
        val_metrics = _classification_metrics(val_true, val_pred)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_metrics["accuracy"],
            "val_acc": val_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "val_f1": val_metrics["f1"],
        }
        history_rows.append(row)

        is_best = False
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            is_best = True

        if epoch_callback is not None:
            payload = dict(row)
            payload["epochs"] = epochs
            payload["is_best"] = is_best
            payload["best_val_f1"] = float(best_f1)
            epoch_callback(payload)

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, pd.DataFrame(history_rows)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    label_names: Sequence[str] = LABEL_NAMES,
) -> Dict[str, object]:
    loss, y_true, y_pred, y_prob = run_epoch(
        model,
        dataloader,
        optimizer=None,
        scheduler=None,
        device=device,
        train_mode=False,
    )
    metrics = _classification_metrics(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=list(label_names),
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)

    return {
        "loss": loss,
        "metrics": metrics,
        "report": report,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def predict_texts(
    model: nn.Module,
    tokenizer,
    texts: Sequence[str],
    device: torch.device,
    max_length: int = 128,
) -> List[Dict[str, object]]:
    model.eval()
    results: List[Dict[str, object]] = []

    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(
                str(text),
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
            pred_id = int(np.argmax(probs))

            results.append(
                {
                    "text": text,
                    "pred_id": pred_id,
                    "label": LABEL_NAMES[pred_id],
                    "confidence": float(probs[pred_id]),
                    "prob_fake": float(probs[0]),
                    "prob_real": float(probs[1]),
                }
            )

    return results


def save_experiment_artifacts(
    model: nn.Module,
    tokenizer,
    output_dir: str,
    model_name: str,
    history_df: pd.DataFrame,
    metadata: Optional[Dict[str, object]] = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    tokenizer.save_pretrained(output_dir)

    history_path = os.path.join(output_dir, "history.csv")
    history_df.to_csv(history_path, index=False)

    payload = {"model_name": model_name}
    if metadata:
        payload.update(metadata)

    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

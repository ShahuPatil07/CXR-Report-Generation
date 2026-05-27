"""
CXR dataset loading, splitting, and PyTorch Dataset class.
"""

import ast
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

INSTRUCTION = "Generate a radiology report for this chest X-ray."

# LLaVA-1.5 chat template
TRAIN_TEMPLATE = "USER: <image>\n{instruction} ASSISTANT: {report}</s>"
EVAL_TEMPLATE  = "USER: <image>\n{instruction} ASSISTANT:"


def _parse_list_col(val):
    if isinstance(val, list):
        return val
    if not isinstance(val, str) or not val.strip():
        return []
    if "|" in val:
        return [x.strip() for x in val.split("|") if x.strip()]
    try:
        return ast.literal_eval(val)
    except Exception:
        return [val]


def _format_report(row) -> str:
    parts = []
    if str(row.get("findings", "")).strip():
        parts.append(f"FINDINGS: {row['findings'].strip()}")
    if str(row.get("impression", "")).strip():
        parts.append(f"IMPRESSION: {row['impression'].strip()}")
    return "\n".join(parts)


def load_csv(csv_path: str | Path) -> pd.DataFrame:
    """
    Load cxr_dataset.csv, parse list columns, build report_text,
    and keep only rows that have ≥1 image and a non-empty report.
    Returns one row per report with a 'primary_image' column pointing
    to the first (PA/frontal) image.
    """
    df = pd.read_csv(csv_path)
    for col in ("mesh_major", "mesh_auto", "image_ids", "image_paths"):
        if col in df.columns:
            df[col] = df[col].fillna("").apply(_parse_list_col)

    df["report_text"] = df.apply(_format_report, axis=1)
    df = df[df["n_images"] > 0].copy()
    df = df[df["report_text"].str.strip() != ""].copy()
    df["primary_image"] = df["image_paths"].apply(lambda x: x[0] if x else None)
    df = df.dropna(subset=["primary_image"]).reset_index(drop=True)
    return df


def load_splits(
    csv_path: str | Path,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split at the report level so PA+Lateral pairs stay together.
    Returns (train_df, val_df, test_df).
    """
    df = load_csv(csv_path)
    tmp_size = int(len(df) * (val_frac + test_frac))
    test_size = int(len(df) * test_frac)

    train_df, tmp_df = train_test_split(df, test_size=tmp_size, random_state=seed)
    val_df, test_df = train_test_split(tmp_df, test_size=test_size, random_state=seed)

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


class CXRDataset(Dataset):
    """
    mode='train': returns processor inputs with labels masked on the
                  instruction (loss computed only on the ASSISTANT response).
    mode='eval':  returns processor inputs ready for model.generate().
    """

    def __init__(self, df: pd.DataFrame, processor, mode: str = "train", max_length: int = 512):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.mode = mode
        self.max_length = max_length

        # Tokenize "ASSISTANT:" once to find the response boundary
        self._assistant_ids = processor.tokenizer.encode(
            "ASSISTANT:", add_special_tokens=False
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["primary_image"]).convert("RGB")

        if self.mode == "train":
            text = TRAIN_TEMPLATE.format(
                instruction=INSTRUCTION, report=row["report_text"]
            )
        else:
            text = EVAL_TEMPLATE.format(instruction=INSTRUCTION)

        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        if self.mode == "train":
            labels = inputs["input_ids"].clone()
            ids = inputs["input_ids"].tolist()
            boundary = _find_subseq(ids, self._assistant_ids)
            if boundary != -1:
                labels[: boundary + len(self._assistant_ids)] = -100
            pad_id = self.processor.tokenizer.pad_token_id
            labels[labels == pad_id] = -100
            inputs["labels"] = labels

        return inputs


def _find_subseq(seq: list[int], sub: list[int]) -> int:
    """Return the starting index of the first occurrence of sub in seq, or -1."""
    n, m = len(seq), len(sub)
    for i in range(n - m + 1):
        if seq[i : i + m] == sub:
            return i
    return -1

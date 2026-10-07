"""Load the dataset from Hugging Face, clean the text, and tag Hinglish rows."""
from datasets import load_dataset

from .preprocess import clean_text, is_hinglish

DATASET = "airzipm/sentiment-dataset-en-hi-hinglish-v2"


def load_data():
    """Returns {"train": df, "validation": df, "test": df} with columns text, label, hinglish."""
    ds = load_dataset(DATASET)
    dfs = {}
    for split in ds:
        df = ds[split].to_pandas()
        df["text"] = df["text"].map(clean_text)
        df["hinglish"] = df["text"].map(is_hinglish)
        dfs[split] = df
    return dfs

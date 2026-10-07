"""Fine-tune a transformer for 3-class sentiment and report Macro F1 (overall / Hinglish / English)."""
import gc
import json
import os
import shutil

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


def run(dfs, name, tag, epochs=2, out_root=".", repo=None,
        batch_size=32, lr=2e-5, seed=42, max_train=None):
    """
    dfs        : output of load_data()
    name       : Hugging Face model name, e.g. "google/muril-base-cased"
    tag        : short name used for output files, e.g. "muril"
    repo       : "username/repo-name" to push the model to the Hub (None = don't push)
    max_train  : use only this many training rows (for a quick smoke test)
    """
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(name)

    def to_ds(df):
        d = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))
        return d.map(lambda b: tok(b["text"], truncation=True, max_length=128), batched=True)

    train_df = dfs["train"]
    if max_train:
        train_df = train_df.sample(max_train, random_state=seed)
    train_ds = to_ds(train_df)
    val_ds = to_ds(dfs["validation"])
    test_ds = to_ds(dfs["test"])

    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=3)

    def metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return {"macro_f1": f1_score(p.label_ids, preds, average="macro")}

    ckpt_dir = os.path.join(out_root, f"{tag}_ckpt")
    args = TrainingArguments(
        output_dir=ckpt_dir, num_train_epochs=epochs, learning_rate=lr,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=64,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="macro_f1",
        fp16=torch.cuda.is_available(), report_to="none", seed=seed,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=metrics, data_collator=DataCollatorWithPadding(tok),
    )
    trainer.train()

    # final test scores (looked at once, after training)
    pred = np.argmax(trainer.predict(test_ds).predictions, axis=1)
    y = dfs["test"].label.values
    m = dfs["test"].hinglish.values
    res = {
        "model": name,
        "overall": float(f1_score(y, pred, average="macro")),
        "hinglish": float(f1_score(y[m], pred[m], average="macro")),
        "english": float(f1_score(y[~m], pred[~m], average="macro")),
    }
    print(res)
    print(classification_report(y[m], pred[m]))

    # save everything right away, before anything can go wrong
    np.save(os.path.join(out_root, f"{tag}_test_preds.npy"), pred)
    with open(os.path.join(out_root, f"{tag}_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    if repo:
        trainer.model.push_to_hub(repo)
        tok.push_to_hub(repo)

    shutil.rmtree(ckpt_dir, ignore_errors=True)   # checkpoints are ~1 GB each
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    return res

# Hinglish Sentiment Classification

Fine-tuned **mBERT** and **MuRIL** for 3-class sentiment (negative / neutral / positive) on a mix of English and
Hinglish (romanized Hindi + English) text, and compared them against a TF-IDF + Logistic Regression baseline.
The focus is on how the models do on the **Hinglish subset**, not just the overall score.

**Demo:** TODO (Hugging Face Space link)  |  **Model:** TODO (Hugging Face Hub link)

## Results (test set, Macro F1)

| Model | Overall | Hinglish | English |
|---|---|---|---|
| TF-IDF + Logistic Regression | 0.817 | 0.518 | 0.826 |
| mBERT | 0.847 | 0.630 | 0.855 |
| MuRIL | TODO | TODO | TODO |

Macro F1 is used because the classes are not balanced, especially on Hinglish (about 58% neutral).

![Confusion matrix (Hinglish)](results/confusion_matrix.png)

## Dataset

[`airzipm/sentiment-dataset-en-hi-hinglish-v2`](https://huggingface.co/datasets/airzipm/sentiment-dataset-en-hi-hinglish-v2)
(about 114K rows; train 96.7K / validation 8.5K / test 8.5K). Labels: 0 = negative, 1 = neutral, 2 = positive.

Things worth knowing:
- It is several datasets merged: movie snippets, IMDB and Yelp reviews, English tweets, and Hinglish tweets.
- About 5% of rows are Hinglish. They were tagged with a simple rule (2 or more common romanized Hindi/Urdu words,
  or any Devanagari). On a manual check of 20 tagged and 20 untagged rows the tag looked right, but it is a heuristic.
- Hinglish rows are almost all tweets, so the "Hinglish" score really means "Hinglish tweets".
- The Hinglish test slice is small (485 rows, 65 positive), so differences of 1-2 points are noise.

## Key findings

TODO: fill in after the error analysis (notebook 03). For example: both transformers improve Hinglish Macro F1 by about
11 points over the baseline; both still over-predict neutral on Hinglish; positive recall is the weakest.

## Repo layout

```
notebooks/
  01_eda_baseline.ipynb     data, Hinglish tagging, TF-IDF baseline
  02_finetune.ipynb         fine-tune MuRIL / mBERT, results table
  03_error_analysis.ipynb   confusion matrix, example mistakes
src/
  preprocess.py             clean_text, is_hinglish
  data.py                   load + clean + tag the dataset
  train.py                  run(): fine-tune, evaluate, save, push to Hub
results/                    metrics and plots
```

## How to run

```bash
pip install -r requirements.txt
```
Run the notebooks in order. Fine-tuning needs a GPU (it was done on Kaggle, 2x T4, about 50 minutes for 2 epochs).
Model weights are not stored in this repo; they are on the Hugging Face Hub.

## Limitations

- Heuristic Hinglish tagging and a small Hinglish test slice.
- Some labels look noisy (for example abusive tweets labeled neutral).
- Texts are truncated to 128 tokens, which cuts long reviews.

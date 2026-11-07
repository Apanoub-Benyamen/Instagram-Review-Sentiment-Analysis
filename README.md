# Instagram App Review Sentiment Analysis

**Python · Natural Language Processing (NLP) · Machine Learning · Deep Learning**

An academic analysis of **10,000 written Instagram app reviews** to explore customer sentiment and compare text-based sentiment classification approaches. The review **text**, not the star rating, is the input used by the sentiment models.

> **Status:** Academic exploratory project. Results from the original notebook have not yet been re-evaluated using a leakage-safe workflow. Reported historical accuracy figures should not be treated as validated model performance.

## Business question

Can app-review text help identify positive, neutral, and negative customer experiences at scale, and highlight opportunities to improve the user experience?

## Dataset

- **10,000 app reviews** collected for an academic project; the original web-scraping script is not included.
- `content`: the written review text used as model input.
- `score`: a 1–5 star rating used as an **approximate reference label** or comparison point, **not** a prediction input.
- The dataset was reduced to these two fields for portfolio use; reviewer names, review IDs, and dates were removed, and recognizable @mentions/email addresses were redacted where detected.

**Important:** The reviews originated on a third-party platform. Please verify that their source terms permit redistributing review text before sharing the CSV publicly. Removing account names does not establish redistribution rights or guarantee that all personal information has been removed. The repository's MIT license applies to original project code and documentation, **not automatically to third-party review content**.

### Rating-based reference labels

| Rating | Proxy class | Records |
|---|---|---:|
| 1–2 stars | Negative | 2,092 (20.92%) |
| 3 stars | Neutral | 392 (3.92%) |
| 4–5 stars | Positive | 7,516 (75.16%) |

![Original star-rating distribution](rating_distribution.png)

*The chart shows the original star ratings, not VADER's independently inferred sentiment from the review text.*

## Methods

1. **Explore and clean text:** inspect review content, convert emojis to words, translate non-English reviews, clean and tokenize text, remove stop words, and lemmatize.
2. **VADER sentiment analysis:** calculate positive/neutral/negative sentiment from review text; compare predictions with rating-derived proxy labels.
3. **Traditional machine learning:** turn review text into TF–IDF features and experiment with Logistic Regression, Linear SVM, Multinomial Naive Bayes, Random Forest, and XGBoost.
4. **Deep learning:** represent reviews as text sequences and experiment with LSTM and Bidirectional LSTM models.

### An important methodological distinction

- **VADER:** reads review text and assigns sentiment without learning from star ratings.
- **Supervised ML / LSTM:** reads review text (`X`) and learns from rating-derived labels (`y`); future predictions use text without requiring a rating.
- Rating-derived labels are **noisy proxies**, so a five-star review can still contain negative language.

## Evaluation limitations

The original exploratory notebook applied TF–IDF and SMOTE **before** splitting data into training and test sets, creating a risk of **data leakage** and optimistic accuracy estimates. The LSTM tokenizer was also fitted on the full text corpus before its split. There is strong class imbalance, with positive ratings representing 75.16% of reviews.

Next steps for rigorous evaluation: split the data first; fit tokenizers/vectorizers only on training data; apply resampling only to training data; compare models using macro F1, per-class precision/recall and confusion matrices. **No historical model accuracy is claimed here as a validated final result.**

## How to explore the notebook

1. Download this repository and install Python/Jupyter (or use Google Colab).
2. Install dependencies with `pip install -r requirements.txt`.
3. If you have permission to use the original reviews, place `Instagram_Reviews_Data.csv` in the **same folder** as `Instagram_Project.ipynb`.
4. Open `Instagram_Project.ipynb` and run its cells in order. The data-loading cell uses `pd.read_csv("Instagram_Reviews_Data.csv")`.
5. Translation steps require internet access, and training deep-learning models may take a long time. The notebook may create a `translated_reviews.csv` checkpoint locally.

**Reproducibility note:** The notebook's saved output cells were removed for safer sharing; it has not been fully rerun with the redacted file and current dependency versions. It illustrates the original academic workflow rather than an independently verified production pipeline.

## Repository files

| File | Purpose |
|---|---|
| `Instagram_Project.ipynb` | Main NLP / ML / LSTM experiment notebook |
| `Instagram_Reviews_Data.csv` | Reduced dataset, **only if redistribution rights permit** |
| `rating_distribution.png` | Rating distribution figure |
| `requirements.txt` | Main Python package dependencies |
| `.gitignore` | Excludes temporary files and checkpoints |
| `LICENSE` | Applies to original code/documentation; not third-party review text |

## Future improvements

- Re-run the models using leakage-safe evaluation and publish corrected metrics.
- Compare text-based sentiment with human-checked labels rather than relying only on ratings.
- Produce a business-facing summary of recurring user feedback themes.

---

**Focus:** Customer feedback analytics · NLP · Model evaluation · Text classification

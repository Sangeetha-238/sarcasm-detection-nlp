# Sarcasm Detection with NLP

An end-to-end Natural Language Processing project that compares classical machine learning and transformer-based approaches for detecting sarcasm in news headlines.

The project progresses from TF-IDF-based baselines to a fine-tuned DistilBERT model, with evaluation, error analysis, reusable inference code, and discussion of dataset limitations.

## Results at a Glance

| Model | Validation Accuracy | Validation F1 |
|---|---:|---:|
| Logistic Regression + TF-IDF | ~85% | ~84% |
| Linear SVM + TF-IDF | ~85% | ~84% |
| DistilBERT | **93.36%** | **93.06%** |

Final DistilBERT test performance:

| Metric | Score |
|---|---:|
| Accuracy | **92.52%** |
| Precision | **90.96%** |
| Recall | **93.56%** |
| F1-score | **92.24%** |
| Test Loss | 0.2690 |

DistilBERT reduced validation classification errors from **648 with Linear SVM to 284**, a reduction of approximately **56%**.

---

## Project Objective

Sarcasm is challenging for NLP systems because the intended meaning of a sentence can differ from its literal wording.

This project investigates whether contextual transformer representations improve sarcasm detection compared with traditional TF-IDF-based machine learning models.

The main goals were to:

- explore and clean a labeled sarcasm dataset,
- build interpretable classical NLP baselines,
- fine-tune a transformer model,
- compare model performance fairly,
- analyze classification errors,
- evaluate generalization limitations,
- and build a reusable prediction pipeline.

---

## Dataset

This project uses the **News Headlines Dataset for Sarcasm Detection** created by Rishabh Misra.

Each record contains:

- `headline` — the news headline text,
- `is_sarcastic` — binary sarcasm label,
- `article_link` — source article URL.

The target classes are:

- `0` — Not Sarcastic
- `1` — Sarcastic

The sarcastic examples are primarily derived from satirical news headlines, while the non-sarcastic examples come from conventional news sources.

Because the dataset file is not committed to this repository, place it locally at:

```text
data/raw/Sarcasm_Headlines_Dataset.json
```

Dataset source:

https://github.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection

---

## Project Workflow

The project follows a complete NLP modeling pipeline:

```text
Raw Dataset
    ↓
Exploratory Data Analysis
    ↓
Cleaning and Deduplication
    ↓
Train / Validation / Test Split
    ↓
TF-IDF Feature Extraction
    ↓
Logistic Regression
    ↓
Linear SVM
    ↓
Classical Model Comparison
    ↓
DistilBERT Fine-Tuning
    ↓
Error Analysis
    ↓
Final Test Evaluation
    ↓
Reusable Prediction Pipeline
```

---

## Repository Structure

```text
sarcasm-detection-nlp/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── svm_model.ipynb
│   ├── 05_model_comparison.ipynb
│   ├── 06_distilbert_model.ipynb
│   └── 07_full_model_comparison.ipynb
│
├── results/
│   ├── logistic_regression_metrics.csv
│   ├── svm_metrics.csv
│   └── distilbert_test_metrics.csv
│
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── features.py
│   ├── evaluate.py
│   └── predict.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Exploratory Data Analysis

The exploratory analysis examines:

- class distribution,
- missing values,
- duplicate headlines,
- headline length,
- word counts,
- sarcastic vs non-sarcastic examples.

Minimal text cleaning was used because punctuation, wording, capitalization, and other surface-level features may contain useful sarcasm signals.

Duplicate headlines were removed before model training.

---

## Data Splitting

The dataset was divided using a stratified split:

- **70% training**
- **15% validation**
- **15% test**

Stratification preserves the sarcastic/non-sarcastic class distribution across all subsets.

The test set remained untouched until final model selection was complete.

---

## Classical NLP Models

### Logistic Regression + TF-IDF

TF-IDF was used to transform headlines into sparse numerical representations using:

- unigrams,
- bigrams,
- a maximum feature limit,
- minimum document frequency filtering.

Logistic Regression achieved approximately:

- **85% validation accuracy**
- **84% F1-score for the sarcastic class**

Feature-coefficient analysis showed that the model learned strong lexical and stylistic patterns associated with the dataset.

---

### Linear SVM + TF-IDF

Linear SVM was trained using the same TF-IDF representation and the same train/validation split.

Validation confusion matrix:

- True Negatives: **1,894**
- False Positives: **348**
- False Negatives: **300**
- True Positives: **1,733**

The SVM made **648 total validation errors**, slightly fewer than Logistic Regression.

Because the difference was small, both classical models performed similarly. This suggests that the feature representation was more influential than the choice between the two linear classifiers.

Linear SVM was selected as the strongest classical baseline.

---

## DistilBERT

A pretrained DistilBERT model was fine-tuned for binary sequence classification.

Unlike TF-IDF, DistilBERT produces contextual word representations, allowing the meaning of each token to depend on the surrounding text.

### Validation Results

| Metric | Score |
|---|---:|
| Accuracy | **93.36%** |
| Precision | **92.51%** |
| Recall | **93.61%** |
| F1-score | **93.06%** |
| Validation Loss | 0.2266 |

Validation confusion matrix:

- True Negatives: **2,088**
- False Positives: **154**
- False Negatives: **130**
- True Positives: **1,903**

Total validation errors: **284**

Compared with Linear SVM, DistilBERT reduced validation errors by approximately **56%**.

---

## Final Test Evaluation

After model selection, DistilBERT was evaluated once on the untouched test set.

### Test Results

| Metric | Score |
|---|---:|
| Accuracy | **92.52%** |
| Precision | **90.96%** |
| Recall | **93.56%** |
| F1-score | **92.24%** |
| Test Loss | 0.2690 |

### Test Confusion Matrix

- True Negatives: **2,054**
- False Positives: **189**
- False Negatives: **131**
- True Positives: **1,902**

The test performance remained close to the validation performance, suggesting that the model generalizes well to unseen examples drawn from the same dataset distribution.

---

## Key Findings

The most important finding is that the largest performance gain came from changing the text representation rather than changing the classical classifier.

Logistic Regression and Linear SVM produced nearly identical results when trained on TF-IDF features.

DistilBERT substantially improved performance by modeling words contextually rather than relying primarily on lexical frequency patterns.

The transformer model:

- improved validation accuracy by roughly eight percentage points,
- reduced validation errors by approximately 56%,
- maintained strong performance on the untouched test set,
- achieved a final test F1-score above 92%.

---

## Error Analysis

Error analysis revealed several recurring challenges.

### False Positives

Some real news headlines were classified as sarcastic because they contained:

- unusual events,
- exaggerated wording,
- unexpected situations,
- or phrasing similar to satirical headlines.

### False Negatives

Some sarcastic headlines were classified as non-sarcastic because the sarcasm depended on:

- background knowledge,
- subtle absurdity,
- implied meaning,
- political or cultural context,
- or information beyond the headline itself.

These results show that sarcasm detection remains difficult even for transformer models.

---

## Dataset Bias and Generalization

Manual inference experiments revealed an important limitation.

The model performs very well on held-out examples from the same dataset, but it does not always generalize reliably to conversational sarcasm or ordinary news-style text.

For example, a conversational sentence such as:

```text
Oh great, another Monday morning.
```

may be classified incorrectly even though the sarcasm is obvious to a human reader.

The model can also assign high confidence to incorrect predictions.

This suggests that it has learned both sarcasm-related signals and source-specific writing patterns.

The reported confidence score is therefore a softmax confidence value and should not be interpreted as a calibrated probability of sarcasm.

---

## Reusable Prediction Pipeline

The final DistilBERT model can be used through:

```text
src/predict.py
```

Example:

```bash
python src/predict.py "Local man shocked to discover actions have consequences"
```

Example output:

```text
Sarcasm Detection Result
------------------------
Text:       Local man shocked to discover actions have consequences
Prediction: Sarcastic
Confidence: 99.85%
```

The inference pipeline:

1. loads the saved tokenizer and model,
2. tokenizes the input text,
3. performs inference,
4. converts logits into softmax scores,
5. returns the predicted class and confidence.

---

## Installation

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd sarcasm-detection-nlp
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Notebooks

Launch Jupyter:

```bash
jupyter notebook
```

Run the notebooks in sequence:

```text
01_data_exploration.ipynb
02_preprocessing.ipynb
03_baseline_model.ipynb
svm_model.ipynb
05_model_comparison.ipynb
06_distilbert_model.ipynb
07_full_model_comparison.ipynb
```

---

## Running Predictions

After training and saving the DistilBERT model locally:

```bash
python src/predict.py "Nation celebrates another completely unnecessary meeting"
```

The trained model files are excluded from GitHub because transformer checkpoints can be large.

---

## Skills Demonstrated

This project demonstrates practical experience with:

- Natural Language Processing
- Text classification
- Exploratory Data Analysis
- Data preprocessing
- TF-IDF
- Logistic Regression
- Linear SVM
- Transformer fine-tuning
- DistilBERT
- Hugging Face Transformers
- PyTorch
- Model evaluation
- Precision, recall, F1-score, and confusion matrices
- Error analysis
- Dataset bias analysis
- Distribution shift
- Reusable Python modules
- Command-line inference
- Git and GitHub project organization

---

## Future Improvements

Potential next steps include:

- training on Reddit or social media sarcasm datasets,
- combining multiple sarcasm datasets,
- evaluating on an external dataset,
- improving cross-domain generalization,
- calibrating prediction confidence,
- experimenting with RoBERTa or DeBERTa,
- using explainability techniques such as SHAP,
- building a Streamlit or Gradio demo,
- publishing the trained model separately,
- and investigating conversational context for sarcasm detection.

---

## Conclusion

This project demonstrates a complete NLP workflow from raw data exploration to reusable model inference.

Classical TF-IDF models provided strong baselines, but the fine-tuned DistilBERT model achieved substantially better performance, reaching **92.52% test accuracy and a 92.24% F1-score**.

At the same time, manual testing revealed that high in-domain performance does not guarantee reliable generalization to every form of sarcasm.

This combination of model performance, error analysis, and limitation analysis highlights both the strengths and practical challenges of applying modern transformer models to sarcasm detection.
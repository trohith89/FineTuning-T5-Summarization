# 🌟 [Hugging Face Deployed Application Link](https://huggingface.co/spaces/trohith89/T5-FineTuning-Summary)

# 📝 KDTS Task 04 — Text Summarization Using NLP


Welcome to the repository for **Task KDT04: Text Summarization Using NLP**, where we fine-tune a transformer-based model to generate high-quality abstractive summaries using Hugging Face Transformers and the CNN/DailyMail dataset.

---

## 🎯 Objective

> Train or fine-tune a transformer model for text summarization using T5, BART, or PEGASUS, and demonstrate the ability to generate concise summaries for long-form articles or news content.

---

## ✅ Requirements Checklist

| Requirement                                                   | Status     |
|---------------------------------------------------------------|------------|
| Use T5, BART, or PEGASUS from Hugging Face                    | ✅ T5-small |
| Train on a small custom dataset or CNN/DailyMail              | ✅ 60K rows from CNN/DailyMail |
| Generate summaries for at least 5 sample paragraphs           | ✅ 10+ samples |
| Compare model-generated summaries vs. extractive method       | ⏳ Coming Soon |
| Include a web form to paste text and view summary             | ⏳ Coming Soon |

---

## 📚 Dataset

We used the [CNN/DailyMail dataset](https://huggingface.co/datasets/cnn_dailymail) and combined 60,000 samples from 3 versions:

- 📘 20K from version `1.0.0`
- 📙 20K from version `2.0.0`
- 📗 20K from version `3.0.0`

**Split Configuration:**
- 🔹 Train: 38,400 samples
- 🔹 Validation: 9,600 samples
- 🔹 Test: 12,000 samples

---

## 🤖 Model Used

We fine-tuned the `t5-small` model for summarization and published it on the Hugging Face Model Hub:

🔗 **Model on Hub**: [trohith89/KDTS_T5_Summary_FineTune](https://huggingface.co/trohith89/KDTS_T5_Summary_FineTune)

---

## 🛠️ Project Pipeline

### 1. Data Preparation  
- Loaded 3 versions of CNN/DailyMail using Hugging Face Datasets
- Combined them into a single DatasetDict
- Split into train/validation/test

### 2. Tokenization  
- Prefixed each article with `summarize:`
- Used `T5TokenizerFast` with truncation and padding

### 3. Model Fine-Tuning  
- Used `Seq2SeqTrainer` from Hugging Face with:
  - Learning rate: `3e-4`
  - Epochs: `3`
  - Batch size: `4`
  - Metric: `rougeL` for best model checkpoint
  - Early stopping and mixed-precision (FP16) enabled

### 4. Inference  
- Generated summaries for test samples
- Compared model summaries with reference summaries

---

## 💡 How to Use the Model

### ✅ Option 1: Hugging Face `pipeline` (Easy Inference)

```python
from transformers import pipeline

pipe = pipeline("text2text-generation", model="trohith89/KDTS_T5_Summary_FineTune")

text = "summarize: The Indian economy grew 6.1% last quarter due to..."
summary = pipe(text)[0]['generated_text']
print("Summary:", summary)


```
✅ Option 2: Load Tokenizer and Model Manually

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("trohith89/KDTS_T5_Summary_FineTune")
model = AutoModelForSeq2SeqLM.from_pretrained("trohith89/KDTS_T5_Summary_FineTune")

input_text = "summarize: India’s economy grew 6.1% last quarter..."
inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
outputs = model.generate(**inputs)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Summary:", summary)

```
```
📈 Evaluation Metrics
We used the following metrics for evaluation:

ROUGE-1

ROUGE-2

ROUGE-L

ROUGE-Lsum

These metrics compare the overlap between the generated and reference summaries.
```
```
KDTS_Task4_Summarization/
├── KDTS_TASK4_FineTuning_Summary_TR.ipynb  # Full training and inference pipeline
├── KDTS-outputs.txt                        # Code outputs from sample runs
├── requirements.txt                        # Required dependencies
├── README.md                               # This documentation
├── app.py                                  # ⏳ Streamlit app (Coming soon)

```
```
🙌 Acknowledgements
🤗 Hugging Face Transformers

📰 CNN/DailyMail Dataset

🧪 KDTS Internship Program Task Series
```

```
📬 Contact
Have questions or feedback?
📧 Reach out via [trohith89@gmail.com]
```
⭐️ Star this repo if you found it useful!

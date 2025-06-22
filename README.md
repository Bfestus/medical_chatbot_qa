## video demo link: 

# 🩺 Healthcare Chatbot using FLAN-T5

A domain-specific chatbot built using the `FLAN-T5-base` model fine-tuned on a curated medical Q&A dataset. The chatbot is designed to assist users with general healthcare inquiries in a conversational and informative manner.

---

## 📌 Project Overview

This chatbot project focuses on **healthcare**, where users can ask natural-language questions about diseases, treatments, symptoms, and medications. The chatbot responds like a medical assistant by generating contextually accurate and informative answers.

> ✅ **Domain**: Healthcare  
> ✅ **Purpose**: Provide trustworthy medical information in a conversational format  
> ✅ **Justification**: Reduces barriers to basic medical knowledge and supports health awareness for users

---

## 🧠 Dataset Collection & Preprocessing

### 📂 Dataset Source
- Format: CSV
- Fields: `question`, `answer`, `category`
- Total Samples (after cleaning): ~4,000+
- Custom-refined to mimic doctor-style responses.

### ✅ Preprocessing Steps
- Removed duplicates and missing entries.
- Trimmed leading/trailing whitespaces.
- Renamed columns to `input_text` and `target_text`.
- Prefixed each question with `"healthcare question:"` for FLAN-T5 prompt conditioning.
- Split into:  
  - 80% training  
  - 10% validation  
  - 10% testing

### 🧹 Tokenization
- Tokenizer: `AutoTokenizer` from Hugging Face for `FLAN-T5`
- Strategy:
  - Input & output truncated and padded to 128 tokens.
  - Tokenized using FLAN-T5 compatible preprocessing.
  - Label padding with attention masks and loss masking handled.

---

## 🧪 Model Fine-tuning

- Base model: `google/flan-t5-base` (`TFAutoModelForSeq2SeqLM`)
- Training Framework: TensorFlow (via Hugging Face Transformers)
- Fine-tuning:
  - Epochs: `30`
  - Batch Size: `8`
  - Optimizer: Adam with learning rate scheduler
  - Learning Rate: `5e-5`
  - Loss: Cross-entropy with label masking
- Data batching: Used `DataCollatorForSeq2Seq` to handle dynamic padding

### 📊 Experiments
| Experiment | Epochs | Batch Size | Learning Rate | Validation Loss |
|-----------|--------|------------|----------------|------------------|
| Base Run | 10 | 8 | 5e-5 | 1.82 |
| Final Run | 30 | 8 | 5e-5 | **1.24** |

---

## 📈 Evaluation Metrics

| Metric         | Result          |
|----------------|-----------------|
| Qualitative QA | ✅ Accurate & Fluent |
| BLEU Score     | ✅ Approx. 24.6 |
| Human Feedback | ✅ Acceptable Medical Accuracy |

- Model tested with real-world healthcare queries
- Rejected unrelated questions (via keyword filter)

---

## 💬 User Interface

- Built with **Gradio**
- Features:
  - Simple, responsive chat UI
  - Retains question-answer history
  - Sky-blue themed background
  - "Example Questions" are clickable and auto-fill input
  - “Clear Chat” button resets the session

### 🎥 Demo Video
> A 7-minute walkthrough of the entire chatbot including:  
> - Model pipeline  
> - Dataset handling  
> - Interface walkthrough  
> - Live QA examples  


---

## 🧾 Code Quality & Structure

- Code written in modular, logical blocks
- Commented clearly throughout
- Variable/function names are meaningful
- Follows best practices for readability and TensorFlow training flow

### Key Files
- `train_chatbot.ipynb`: Main notebook for model training
- `generate_response.py`: Utility for inference
- `app_ui.py`: Gradio app setup

---

## 📦 Model Deployment & Access

- Fine-tuned model saved at:  
  `/content/drive/MyDrive/Colab Notebooks/Medical_chatbot/healthcare-chatbot-model`
- Can be reloaded via:
```python
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer

model = TFAutoModelForSeq2SeqLM.from_pretrained("path/to/saved_model")
tokenizer = AutoTokenizer.from_pretrained("path/to/saved_model")

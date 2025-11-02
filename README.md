#  Surgical Intelligence Pipeline

A cost-optimized AI pipeline for automatic German medical audio transcription and summarization.
The project evaluates multiple open-source and commercial ASR models and free/low-cost LLMs to identify a reliable alternative to AWS Transcribe for healthcare data workflows.

This system converts multilingual surgical or clinical recordings into structured text and concise summaries through integrated ASR and LLM modules.

---

##  Project Objective

The primary objective of this project is to find and evaluate efficient, low-cost AI models capable of:

1. Transcribing German medical speech accurately into text using ASR systems.

2. Summarizing long-form medical transcripts using free or lightweight open LLMs.

3. Providing a scalable and budget-friendly alternative to enterprise services such as AWS Transcribe for internal medical documentation pipelines.

---

##  Technical Overview

### 1. **Audio → Text (Automatic Speech Recognition)**
Benchmarked multiple **ASR models** for multilingual surgical audio transcription:

| Model | Type | Key Features |
|:------|:------|:-------------|
| **OpenAI Whisper (Tiny–Medium–Large)** | Transformer ASR | High accuracy, multilingual support |
| **AssemblyAI** | API ASR | Strong WER performance, stable timestamps |
| **Salad Transcription API** | Cloud ASR | Fast inference, low latency |
| **Vosk** | Offline ASR | Lightweight and open-source |
| **AWS Transcribe** | Cloud ASR | Baseline reference model |

**Evaluation Metrics:**  
`BLEU`, `ROUGE`, `METEOR`, `Word Error Rate (WER)`, `Character Error Rate (CER)`, `Match Error Rate (MER)`, and `Word Information Loss (WIL)`.

---

### 2. **Text Cleaning and Translation**
Pre- and post-processing of transcripts to improve model readability:
- Used **TextCy** and **NLTK** for cleaning stopwords and non-medical tokens.  
- Translated German transcripts into English using:  
  - `icky/translate` (Gemma2-based)  
  - `dorian2b/vera` (Qwen-based)  
  - `Helsinki-NLP/opus-mt-de-en` (Transformer)  
- Implemented **chunking** strategies to handle long documents efficiently.

---

### 3. **Summarization and Prompt Engineering**
Designed a **two-round LLM summarization framework**:
1. **Round 1:** Summarize segmented chunks using instruction-tuned prompts.  
2. **Round 2:** Merge and refine outputs into a single coherent summary.  

Tested multiple **Large Language Models**:
- `GPT-4` (OpenAI)
- `Gemini-2 Flash` (Google)
- `DeepSeek-R1`
- `Qwen-2`
- `Gemma-2`

All prompts were designed with **temperature=0** for deterministic, extractive summaries.

---

### 4. **Keyword & Entity Extraction**
Integrated **Named Entity Recognition (NER)** and unsupervised keyword extraction:
- Models:  
  - `Clinical-AI-APOLLO/MedicalNERExtractor`  
  - `BioClinicalBERT`  
  - `SciSpaCy`  
  - `HUMADEX/english-medical-ner`  
- Applied **LDA** (Latent Dirichlet Allocation) for thematic keyword grouping.  
- Combined results with **GPT-based post-processing** to ensure domain relevance.

---

### 5. **Timestamp and Step Extraction**
Developed and evaluated a **prompt-based timestamp extraction pipeline**:
- Used semantic comparison (cosine similarity, F1-scores on entities) to align predicted vs. manual steps.
- Integrated **ScispaCy** and **Sentence Transformers** for semantic validation.

---

## 📊 Dataset

For testing and evaluation, the project used:
- A **public German medical speech dataset** from **FutureBeeAI**, consisting of *30 short .wav files* (≈4 minutes total) with verified transcripts — used to benchmark transcription and translation performance.

---

## 🧠 Tech Stack

| Category | Tools / Libraries |
|:----------|:------------------|
| **ASR** | Whisper, AssemblyAI, Vosk, Salad API |
| **NLP / NER** | SpaCy, SciSpaCy, TextCy, NLTK, Transformers |
| **LLMs** | GPT-4, Gemini-2 Flash, DeepSeek-R1, Qwen-2, Gemma-2 |
| **Evaluation** | ROUGE, BLEU, METEOR, JiWER, Scikit-Learn |
| **Data Handling** | Pandas, NumPy, JSON, Requests, PyDub |
| **Visualization & Analysis** | Matplotlib, Sentence-Transformers |

---

## 🧪 Repository Structure

```
surgical-intelligence-pipeline/
│
├── data/
│
│
├── notebooks/
│   ├── 01_asr_transcription_benchmark.ipynb
│   ├── 02_text_cleaning_and_translation_pipeline.ipynb
│   ├── 03_german_dataset_evaluation.ipynb
│   ├── 04_medical_entity_and_keyword_extraction.ipynb
│   ├── 05_llm_prompted_summarization_pipeline.ipynb
│   ├── 06_timestamp_alignment_evaluation.ipynb
│   └── 07_timestamp_extraction_pipeline.ipynb
```

---

## 🚀 Key Technical Contributions
- Designed and implemented **a modular multilingual ASR–LLM pipeline**.  
- Benchmarked multiple speech-to-text models using custom evaluation metrics.  
- Engineered **prompt templates** for medical summarization and timestamp extraction.  
- Integrated **NER + topic modeling + semantic similarity** validation for accuracy control.  
- Built a scalable structure for future integration into healthcare AI systems.

---

## ⚙️ How to Run
```bash
# 1. Clone the repository
git clone https://github.com/<yourusername>/surgical-intelligence-pipeline.git
cd surgical-intelligence-pipeline

# 2. Run notebooks in sequence
jupyter notebook notebooks/
```

---

## 🧾 License
This project is released for educational and research demonstration purposes under the **MIT License**.  
All sensitive datasets and company-specific materials have been removed.

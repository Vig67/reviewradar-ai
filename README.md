# reviewradar-ai

An individual capstone project demonstrating the deployment of production-grade Artificial Intelligence for advanced data interpretation, text classification, and business intelligence insights.

## 🔗 Live Application Link
live app : https://reviewradar-ai-ejvt4o7wzwuoidnq8u4k5y.streamlit.app/

---

## 📌 Project Overview & Business Value
In modern data ecosystems, unstructured textual data (customer reviews, support tickets, social media mentions) contains vital operational signals. Traditional keyword-matching architectures fail to understand human nuance, sarcasm, and conditional context.

**Insightpulse** addresses this issue by utilizing an advanced Deep Learning Transformer model to automate **text classification**. The system ingests raw consumer reviews via a real-time playground or a bulk CSV upload pipeline, structures the data through programmatic analysis, and maps out actionable insights via an interactive data visualization layer.

---

## 🎯 Core Competencies Demonstrated
This project serves as the final deliverable for my data analytics curriculum and highlights four fundamental pillars:
* **Data Interpretation using AI:** Moving past raw numeric data frames to translate model output into strategic business summaries.
* **Analytical Thinking:** Engineering data filtering pathways to uncover *why* specific sentiment trends or operational anomalies occur.
* **AI-Assisted Insights Generation:** Utilizing modern NLP models to programmatically extract keywords and contextual sentiment tags.
* **Individual Project Ownership:** Managing the full-stack software lifecycle—from environment provisioning and coding to public cloud deployment.

---

## 🛠️ Technical Concepts & Architecture
The tool uses a clean modular architecture to separate data ingestion, model inference, and information visualization layers:

1.  **Sentiment Analysis Fundamentals:** Translating string datasets into token patterns to determine positive and negative emotional weights.
2.  **Text Classification (NLP):** Leveraging **DistilBERT** (a light, optimized variant of Google's BERT architecture) for high-accuracy semantic sequencing.
3.  **Basic Machine Learning Concepts:** Utilizing fine-tuned model pipelines, probability thresholds, confidence matrices, and cached resource states.
4.  **Data Visualization Basics:** Building automated UI metric cards and reactive frequency distribution bar charts via Pandas and Streamlit.

---

## 🎓 Academic Foundation (Coursera)
The technical foundation for this system was cultivated across two key specializations:
* **Python for Data Science, AI & Development (IBM):** Provided core mastery over data manipulation vectors, file I/O operations, and standard object structures (`Pandas`, `Numpy`).
* **Generative AI with Large Language Models (AWS & DeepLearning.AI):** Instructed the engineering principles of context windows, transformer layers, text embeddings, and model operational efficiencies.

---

## 📁 Repository Structure
```text
├── app.py                  # Main Streamlit web application engine
├── requirements.txt        # Production dependencies mapped for the hosting server
├── README.md               # Executive project documentation (This file)
└── sentiment_test_dataset.csv # Standard evaluation matrix file used for validation testing

# Hyperlocal-News-Anomaly-Detection
📌 Project Overview
This project builds an end-to-end NLP system designed to detect anomalous patterns in hyperlocal news articles, identify possible misattribution of news sources, and monitor narrative evolution over time.
Hyperlocal news often contains subtle cues about geographical context, writing style, regional topics, and sentiment. When an article’s content does not match its reported publication location, it may indicate misinformation, manipulated narratives, or cross-regional misreporting.

This system automatically:
•	Extracts locations from unstructured text
•	Analyzes linguistic, thematic, and sentiment-based signals
•	Performs anomaly detection
•	Assesses content–origin consistency
•	Flags unusual or misleading news events
•	Supports scalable deployment on GCP or AWS
________________________________________
✨ Key Features

1. Location Extraction (NER & Geocoding)
•	Named Entity Recognition using spaCy/Flair/Transformer-based models
•	Rule-based fallback for ambiguous mentions
•	Geocoding via Google Maps API / Nominatim
•	Multi-location scoring for robust inference

2. Content–Origin Consistency Check
The system compares:
•	Extracted location vs. claimed publication location
•	Localized keywords, entities, and cultural markers
•	Dialectal or linguistic patterns
•	Local sentiment-topic alignment
Outputs a consistency score ∈ [0,1]

3. NLP-Based Feature Extraction
•	TF-IDF or BERT embeddings
•	Topic modeling (LDA/BERT-based topics)
•	Sentiment & emotion scoring
•	Stylistic signatures (n-grams, POS patterns, writing style vectors)
4. Anomaly Detection
•	Isolation Forest
•	LOF
•	Autoencoder-based reconstruction error
•	Statistical outlier scoring on embeddings

Flags articles with:
✔ unusual topics for region
✔ extreme sentiment shifts
✔ structurally abnormal writing
✔ mismatch between content & inferred origin
5. Narrative Drift Monitoring

Tracks evolving themes across:
•	Time series analyses
•	Topic shifts
•	Changes in local discourse patterns
Useful for misinformation tracking and hyperlocal media monitoring.
________________________________________
🧱 System Architecture
          ┌──────────────────────┐
          │  Raw News Articles   │
          └──────────┬───────────┘
                     ▼
        ┌────────────────────────────┐
        │ 1. Preprocessing & Cleaning│
        └──────────┬─────────────────┘
                   ▼
     ┌───────────────────────────────┐
     │ 2. Location Extraction (NER)  │
     └──────────┬────────────────────┘
                ▼
    ┌─────────────────────────────────┐
    │ 3. NLP Feature Engineering      │
    └────────────┬────────────────────┘
                 ▼
   ┌──────────────────────────────────┐
   │ 4. Anomaly Detection Models      │
   └───────────────┬─────────────────┘
                   ▼
   ┌──────────────────────────────────┐
   │ 5. Content-Location Consistency  │
   └───────────────┬─────────────────┘
                   ▼
      ┌──────────────────────────────┐
      │ Alerts / Dashboard / API     │
      └──────────────────────────────┘
________________________________________
🛠️ Tech Stack
NLP & ML
•	Python (spaCy, NLTK, HuggingFace Transformers)
•	Scikit-learn, PyTorch/TensorFlow
•	Topic modeling (LDA, BERTopic)
Backend / APIs
•	FastAPI / Flask
•	REST API for article submission & anomaly scoring
Data Storage
•	PostgreSQL / BigQuery (GCP)
•	DynamoDB / RDS (AWS)
Deployment Options
AWS
•	Lambda + API Gateway (serverless)

•	DynamoDB/RDS for metadata
________________________________________

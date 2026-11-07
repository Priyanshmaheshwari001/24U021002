# Humour Detection in Social Media

## Overview
This project is an NLP course project focusing on **Humour Detection in Social Media** using Large Language Model (LLM) APIs (zero-shot / few-shot prompting, without model fine-tuning or training). The task is framed as binary text classification:
- **`1`**: Funny / Humorous
- **`0`**: Not Funny / Non-humorous

## Project Structure
```text
humour-detection-api/
├── app/                  # Interactive demo application (Streamlit)
├── data/
│   ├── raw/              # Original raw datasets
│   └── processed/        # Cleaned and preprocessed datasets
├── report/               # Project report, documentation, and write-ups
├── results/
│   ├── figures/          # Evaluation plots and visual artifacts
│   └── predictions/      # LLM inference outputs and evaluation metrics
├── src/                  # Core source code (preprocessing, prompting, eval)
├── .env.example          # Template for environment variables (API keys)
├── .gitignore            # Files and directories ignored by Git
├── README.md             # Project overview and documentation
└── requirements.txt      # Project Python dependencies
```

## Setup Instructions

### 1. Environment Setup
Create and activate a virtual environment:
```bash
python -m venv venv
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# Linux / macOS
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. API Configuration
Copy `.env.example` to `.env` and provide your Google Gemini API key:
```bash
cp .env.example .env
```
Update `.env`:
```env
GEMINI_API_KEY="your_actual_gemini_api_key"
```

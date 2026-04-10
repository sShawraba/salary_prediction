# 📊 Salary Prediction Application (End-to-End ML Pipeline)

## 🚀 Overview
This project is an end-to-end machine learning system that predicts data science job salaries based on job-related features such as experience level, employment type, job title, company size, and location.

It also includes:
- A trained ML model (Decision Tree / Random Forest baseline)
- A FastAPI service for predictions
- A Streamlit dashboard for interactive exploration and storytelling
- Optional LLM-based insights generation using a local model (Ollama)

---

## 🧠 Project Goal
To understand salary patterns in data science roles and build a system that:
1. Predicts expected salary from job features
2. Explains predictions using data analysis and visualization
3. Provides an interactive interface for exploration

---

## 🏗️ Architecture

Raw Data → Preprocessing → ML Model → FastAPI → Streamlit Dashboard
                          ↘
                           LLM (Ollama for insights)

---

## 📁 Project Structure

salary_prediction/
├── data/
│   ├── raw/
│   └── cleaned/
├── src/
│   ├── models/
│   ├── routers/
│   ├── llm/
│   └── data_processing.py
├── dashboard/
│   └── app.py
├── main.py
├── requirements.txt
└── README.md

---

## ⚙️ Setup Instructions

### 1. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
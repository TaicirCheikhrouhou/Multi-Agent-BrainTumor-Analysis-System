# 🧠 BrainTumorAISystem

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-v1.x-yellow)](https://streamlit.io/)
[![CrewAI](https://img.shields.io/badge/CrewAI-agent‑pipeline‑brightgreen)](https://crewai.dev/)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

## 🚀 Project Overview

**BrainTumorAISystem** is an AI-powered multi-agent system for automated brain tumor analysis using MRI images. It orchestrates specialized agents to classify tumors, provide clinical insights, recommend treatments, and generate medical reports—all through an intuitive Streamlit interface. Strengths include modular design, seamless integration of ML and LLMs, and privacy-focused local execution.

---

## 🧩 Key Features

✔ **Accurate Classification**: TensorFlow VGG19 model for reliable tumor detection  
✔ **Intelligent Agents**: CrewAI-powered pipeline for clinical analysis and recommendations  
✔ **Comprehensive Reports**: Structured medical documentation with downloadable outputs  
✔ **User-Friendly UI**: Streamlit dashboard for easy image uploads and result visualization  
✔ **Optional Graph Integration**: Neo4j-based medical knowledge graph for advanced insights  
✔ **Privacy & Speed**: Local LLM execution ensures data security and fast processing

---

## 🛠 Architecture

```
MRI Upload (Streamlit)
    ↓
Classification Agent (VGG19)
    ↓
Clinical + Recommendation Agents
    ↓
Report Agent (Final Summary)
    ↓
Results + Downloadable Report
```

---

## ⚡ Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/BrainTumorAISystem.git
cd BrainTumorAISystem
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Pull and run the local LLM
```bash
ollama pull mistral:latest
ollama serve
```

### 5. Place the model
Put your trained VGG19 model in:
```
models/best_model_VGG19.keras
```

### 6. Run the app
```bash
streamlit run app.py
```

## 🧠 Neo4j Integration (Optional)
To enable medical knowledge graph features:
- Install Neo4j Desktop or Server
- Configure connection in [`neo4j_connector.py`](neo4j_connector.py )
- Use sidebar buttons to test and render the graph

## 🧪 Example Output
*(Add screenshots or animated walkthroughs here)*  
Upload → Agent Progress → Results per agent → Final medical report

## 📦 Folder Structure
```
├── crew/                     # CrewAI agents + task definitions
├── models/                   # Model artifacts
├── tools/                    # Tool modules used by agents
├── config.py                 # LLM and environment configuration
├── app.py                    # Streamlit interface
├── neo4j_connector.py        # DB connector
├── neo4j_visualizer.py       # Graph rendering
└── requirements.txt          # Dependencies
```

## 🙌 Contributing
We welcome contributions! Feel free to open issues or propose enhancements.
# AI/Data Portfolio — AUEB (AI Data Factory)

This repository contains projects developed as part of courses in the **AI Data Factory** program at the **Athens University of Economics and Business (AUEB)**

## Overview
- Projects are **grouped by category** (e.g., Supervised Learning, Unsupervised Learning, Deep Learning, Recommender Systems).

- Each project folder includes:
	-`notebook.ipynb` → complete project workflow with results, visualizations, and explanations (in **Greek**).
	- `main.py` & `utils.py` → modular **production-ready** code (comments and docstrings in English).

- The `requirements/` folder contains a global `requirements.txt` file listing all Python dependencies used across projects.
	- It is intentionally broad to cover multiple environments and assignments.

## Execution (General Instructions)
1) Create a Python environment (e.g., venv).
2) Install the dependencies:
   ```
   pip install -r requirements/requirements.txt
   ```
3) Navigate to the project folder and run:
   ```
   python main.py
   ```
   Adjust dataset paths or configuration settings as needed (see each project's README for details).
   
## Structure
```
repo-root/
│
├─ requirements/
│   └─ requirements.txt
│
├─ supervised_pr/
│   ├─ project_1/
│   │   ├─ notebook.ipynb
│   │   ├─ main.py
│   │   └─ utils.py
│   └─ ...
│
├─ unsupervised_pr/
│   ├─ project_1/
│   │   ├─ notebook.ipynb
│   │   ├─ main.py
│   │   └─ utils.py
│   └─ ...
│
├─ dlnlp/
│   ├─ project_1/
│   │   ├─ notebook.ipynb
│   │   ├─ main.py
│   │   └─ utils.py
│   └─ ...
│
└─ recommend/
    ├─ project_1/
    │   ├─ notebook.ipynb
    │   ├─ main.py
    │   └─ utils.py
    └─ ...
```

## Notes
The goal of this repository is to demonstrate **clean, modular code** (`main.py` / `utils.py`) alongside **full analytical workflows** (`notebook.ipynb`).
Datasets are not always included; please refer to each project’s folder for details or download instructions.
All **notebooks are in Greek**, while all **Python code files are commented in English** for clarity and international readability.
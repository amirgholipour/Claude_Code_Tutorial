---
allowed-tools: Bash(python app.py:*), Bash(uv pip install:*), Bash(cd:*), Bash(python --version:*), Bash(uv pip list:*), Bash(ls:*)
description: Install dependencies and launch the System Design Interview Masterclass Gradio app for Lesson 3
---

## Context

- Python version: !`python --version 2>&1`
- uv version: !`uv --version 2>/dev/null || echo "not installed"`
- App file exists: !`ls lessons/03-system-design-interview/app/app.py 2>/dev/null || echo "NOT FOUND — run from project root"`
- Gradio installed: !`python -c "import gradio; print('gradio', gradio.__version__)" 2>/dev/null || echo "not installed"`
- Plotly installed: !`python -c "import plotly; print('plotly', plotly.__version__)" 2>/dev/null || echo "not installed"`
- Scikit-learn installed: !`python -c "import sklearn; print('sklearn', sklearn.__version__)" 2>/dev/null || echo "not installed"`

## Your task

Launch the System Design Interview Masterclass Gradio app.

1. Check if `lessons/03-system-design-interview/app/app.py` exists. If not, tell the user: "Run from the project root directory (`C:\Saeed\Projects\Claude_Code_Tutorial`)."

2. Check if gradio, plotly, and sklearn are installed. If any are missing:
   - If `uv` is not installed: `pip install uv`
   - Then install: `uv pip install -r lessons/03-system-design-interview/app/requirements.txt`
   - This takes 1–3 minutes with uv (no large packages like PyTorch — much faster than Lesson 2)

3. Launch the app:
   ```
   cd lessons/03-system-design-interview/app && python app.py
   ```

4. Tell the user: "App is running at http://localhost:7862 — open in your browser."

5. If the app fails to start:
   - Show the error message
   - Common fixes:
     - Port in use: change `server_port` in app.py (currently 7862) to any free port (e.g., 7863)
     - Import error for a module: this is expected if modules/m0X_*.py files don't exist yet — the app degrades gracefully and shows an error message in that tab only
     - `rank_bm25` missing: `uv pip install rank_bm25`
     - `google-generativeai` missing: this is optional — app works without it; install with `uv pip install google-generativeai`

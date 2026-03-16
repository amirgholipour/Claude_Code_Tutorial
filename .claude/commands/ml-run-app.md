---
allowed-tools: Bash(python app.py:*), Bash(uv pip install:*), Bash(cd:*), Bash(python --version:*), Bash(uv pip list:*), Bash(ls:*)
description: Install dependencies and launch the ML & Deep Learning Gradio app for Lesson 2
---

## Context

- Python version: !`python --version 2>&1`
- uv version: !`uv --version 2>/dev/null || echo "not installed"`
- App file exists: !`ls lessons/02-ml-with-gradio/app/app.py 2>/dev/null || echo "NOT FOUND — run from project root"`
- Gradio installed: !`python -c "import gradio; print('gradio', gradio.__version__)" 2>/dev/null || echo "not installed"`
- PyTorch installed: !`python -c "import torch; print('torch', torch.__version__)" 2>/dev/null || echo "not installed"`
- Scikit-learn installed: !`python -c "import sklearn; print('sklearn', sklearn.__version__)" 2>/dev/null || echo "not installed"`

## Your task

Launch the ML & Deep Learning Gradio app.

1. Check if `lessons/02-ml-with-gradio/app/app.py` exists. If not, tell the user: "Run from the project root directory (`C:\Saeed\Projects\Claude_Code_Tutorial`)."

2. Check if gradio, torch, and sklearn are installed. If any are missing:
   - If `uv` is not installed: `pip install uv`
   - Then install: `uv pip install -r lessons/02-ml-with-gradio/app/requirements.txt`
   - This takes 2–5 minutes with uv (much faster than pip; PyTorch is ~2 GB download)

3. Launch the app:
   ```
   cd lessons/02-ml-with-gradio/app && python app.py
   ```

4. Tell the user: "App is running at http://localhost:7861 — open in your browser."

5. If the app fails to start:
   - Show the error message
   - Common fixes:
     - Port in use: change `server_port` in app.py (currently 7861) to any free port
     - Import error: check which module failed and suggest `uv pip install <package>`
     - CUDA error: PyTorch CPU-only install → `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

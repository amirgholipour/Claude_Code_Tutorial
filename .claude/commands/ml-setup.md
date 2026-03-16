---
allowed-tools: Bash(uv pip install:*), Bash(uv pip list:*), Bash(python -c:*), Bash(uv --version:*)
description: Install all Python dependencies for Lesson 2 ML & Deep Learning app
---

## Context

- Requirements file: !`cat lessons/02-ml-with-gradio/app/requirements.txt 2>/dev/null || echo "NOT FOUND"`
- Installed packages (relevant): !`uv pip list 2>/dev/null | grep -E "gradio|torch|sklearn|plotly|numpy|pandas|shap" || echo "uv not available"`
- Python version: !`python --version 2>&1`
- uv version: !`uv --version 2>/dev/null || echo "not installed — run: pip install uv"`
- Disk space: !`df -h . 2>/dev/null | tail -1 || echo "unknown"`

## Your task

Install all dependencies for the Lesson 2 ML & Deep Learning Gradio app using `uv` (fast Rust-based installer).

1. Check `requirements.txt` is found. If not: tell user to run from project root.

2. Check `uv` is available. If not, install it first:
   ```
   pip install uv
   ```

3. Create a virtual environment (recommended):
   ```
   python -m venv .venv
   ```
   Activate it:
   - Windows: `.venv\Scripts\activate`
   - macOS/Linux: `source .venv/bin/activate`

4. Install requirements with uv (much faster than pip):
   ```
   uv pip install -r lessons/02-ml-with-gradio/app/requirements.txt
   ```

   **Note on PyTorch:**
   - Default install: ~2 GB download, CUDA enabled (if GPU available)
   - For CPU-only (smaller, faster install):
     ```
     uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
     ```
   - Ask the user: "Do you have an NVIDIA GPU? If yes, install full PyTorch. If no, CPU-only is faster to download."

5. Verify all key packages after install:
   ```python
   python -c "
   import gradio; print('gradio', gradio.__version__)
   import sklearn; print('sklearn', sklearn.__version__)
   import torch; print('torch', torch.__version__)
   import plotly; print('plotly', plotly.__version__)
   import shap; print('shap', shap.__version__)
   import scipy; print('scipy', scipy.__version__)
   "
   ```

6. If any package fails to verify, show the specific error and suggest: `uv pip install <package>` individually.

7. When all verified: "Setup complete! Run `/ml-run-app` to start the course."

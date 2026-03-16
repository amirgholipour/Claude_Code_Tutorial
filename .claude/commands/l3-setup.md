---
allowed-tools: Bash(uv pip install:*), Bash(uv pip list:*), Bash(python -c:*), Bash(uv --version:*)
description: Install all Python dependencies for Lesson 3 System Design Interview app
---

## Context

- Requirements file: !`cat lessons/03-system-design-interview/app/requirements.txt 2>/dev/null || echo "NOT FOUND"`
- Installed packages (relevant): !`uv pip list 2>/dev/null | grep -E "gradio|plotly|sklearn|numpy|pandas|scipy|rank_bm25|google" || echo "uv not available"`
- Python version: !`python --version 2>&1`
- uv version: !`uv --version 2>/dev/null || echo "not installed — run: pip install uv"`
- Disk space: !`df -h . 2>/dev/null | tail -1 || echo "unknown"`

## Your task

Install all dependencies for the Lesson 3 System Design Interview Masterclass Gradio app using `uv` (fast Rust-based installer).

1. Check `requirements.txt` is found. If not: tell user to run from project root (`C:\Saeed\Projects\Claude_Code_Tutorial`).

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
   uv pip install -r lessons/03-system-design-interview/app/requirements.txt
   ```

   **Note on google-generativeai:**
   - This package is optional — the app works fully without it using simulated responses
   - If you want live Gemini API responses, this package is required
   - Get a free API key at: https://aistudio.google.com/app/apikey

5. Verify all key packages after install:
   ```python
   python -c "
   import gradio; print('gradio', gradio.__version__)
   import plotly; print('plotly', plotly.__version__)
   import sklearn; print('sklearn', sklearn.__version__)
   import numpy; print('numpy', numpy.__version__)
   import pandas; print('pandas', pandas.__version__)
   import scipy; print('scipy', scipy.__version__)
   try:
       import rank_bm25; print('rank_bm25 OK')
   except ImportError:
       print('rank_bm25 MISSING — run: uv pip install rank_bm25')
   try:
       import google.generativeai; print('google-generativeai OK')
   except ImportError:
       print('google-generativeai not installed (optional — simulated mode works without it)')
   "
   ```

6. If any required package fails to verify, show the specific error and suggest: `uv pip install <package>` individually.

7. When all verified: "Setup complete! Run `/l3-run` to start the System Design Interview Masterclass."

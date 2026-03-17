"""
System Design Interview Masterclass
Run: python app.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import gradio as gr

# ── Import all modules — failures are caught individually below ────────────────
_import_errors = {}


def _try_import(name):
    try:
        mod = __import__(f"modules.{name}", fromlist=[name])
        return mod
    except Exception as e:
        _import_errors[name] = str(e)
        return None


# ── Module imports ─────────────────────────────────────────────────────────────
m01 = _try_import("m01_data_model_design")
m02 = _try_import("m02_data_architecture")
m03 = _try_import("m03_ml_system_design")
m04 = _try_import("m04_rag_system_design")
m05 = _try_import("m05_rag_security")
m06 = _try_import("m06_agentic_ai_design")
m07 = _try_import("m07_multi_agent_design")


def _build_tab_safe(module, module_name: str):
    """Call module.build_tab() inside a try/except; show error message on failure."""
    if module is None:
        err = _import_errors.get(module_name, "Unknown import error")
        gr.Markdown(
            f"**Module `{module_name}` failed to load.**\n\n"
            f"```\n{err}\n```\n\n"
            "Check that all dependencies are installed:\n"
            "```bash\npip install -r requirements.txt\n```"
        )
        return

    try:
        module.build_tab()
    except Exception as e:
        import traceback
        gr.Markdown(
            f"**Error building tab for `{module_name}`:**\n\n"
            f"```\n{traceback.format_exc()}\n```"
        )


# ── App ────────────────────────────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft(), title="System Design Interview Masterclass") as demo:
    gr.Markdown("""
# 🏛️ System Design Interview Masterclass
**v1.0.0 — Comprehensive Interview Prep for Google, Meta, Amazon, Microsoft, Nvidia**

> 7 modules covering data modeling, ML systems, RAG pipelines, security, and agentic AI.
> Optional: connect a Gemini API key for live LLM-powered feedback.

| Level | Modules |
|---|---|
| 🟡 Intermediate | Data Model Design, Data Architecture, ML System Design |
| 🔴 Advanced | RAG Design, RAG Security, Agentic AI, Multi-Agent Systems |
""")

    with gr.Tabs():
        with gr.Tab("🗄️ 01 · Data Model Design"):
            _build_tab_safe(m01, "m01_data_model_design")

        with gr.Tab("🏗️ 02 · Data Architecture"):
            _build_tab_safe(m02, "m02_data_architecture")

        with gr.Tab("🤖 03 · ML System Design"):
            _build_tab_safe(m03, "m03_ml_system_design")

        with gr.Tab("🔍 04 · RAG Design"):
            _build_tab_safe(m04, "m04_rag_system_design")

        with gr.Tab("🔒 05 · RAG Security"):
            _build_tab_safe(m05, "m05_rag_security")

        with gr.Tab("⚡ 06 · Agentic AI"):
            _build_tab_safe(m06, "m06_agentic_ai_design")

        with gr.Tab("🕸️ 07 · Multi-Agent"):
            _build_tab_safe(m07, "m07_multi_agent_design")


if __name__ == "__main__":
    demo.launch(server_port=7862, server_name="127.0.0.1",
                share=False, show_error=True)

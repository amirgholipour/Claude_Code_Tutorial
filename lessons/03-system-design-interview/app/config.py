"""
App configuration for System Design Interview Masterclass
"""

APP_TITLE = "System Design Interview Masterclass"
APP_DESCRIPTION = "Comprehensive interview prep for Google, Meta, Amazon, Microsoft, Nvidia — 7 modules covering data, ML, RAG, and Agentic AI systems"
APP_VERSION = "1.0.0"

COLORS = {
    "primary": "#6366F1",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "info": "#3B82F6",
    "palette": [
        "#6366F1", "#10B981", "#F59E0B", "#EF4444", "#3B82F6",
        "#8B5CF6", "#06B6D4", "#F97316", "#84CC16", "#EC4899",
    ],
}

GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3-flash",
    "gemini-3.1-pro",
]

LLM_PROVIDERS = ["Local (Simulated)", "Google Gemini"]

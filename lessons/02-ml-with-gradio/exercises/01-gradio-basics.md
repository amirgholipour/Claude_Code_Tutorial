# Exercise 1: Gradio Basics

## Goal

Install Gradio and build your first interactive ML app from scratch — understanding the core building blocks before exploring the full course app.

## What You'll Learn

- What Gradio is and why it's used for ML demos
- Core Gradio components: Textbox, Slider, Dropdown, Button, Plot
- How event handlers (`.click()`, `.change()`) wire inputs to functions
- How to launch a Gradio app locally

## Background

**Gradio** is a Python library that turns any function into a shareable web interface in minutes. Instead of building HTML/CSS/JS, you describe inputs and outputs in Python — Gradio handles the rest.

```
Python function:  f(x) → y
Gradio wraps it: Slider(x) → [Run button] → Plot(y)
```

It's the standard tool for ML demos because:
- Zero frontend knowledge required
- Live reload during development
- Built-in support for images, audio, DataFrames, and Plotly charts
- One-line sharing via Gradio's public tunneling (`share=True`)

## Steps

### Step 1: Install Gradio

```bash
cd lessons/02-ml-with-gradio/app
pip install -r requirements.txt
```

Or install Gradio alone:
```bash
pip install gradio
```

Verify:
```python
import gradio as gr
print(gr.__version__)   # Should be 4.x or higher
```

### Step 2: Your first app (2 minutes)

Create a file `hello_ml.py` anywhere and paste:

```python
import gradio as gr

def greet(name, enthusiasm):
    return f"Hello, {name}!" + "!" * int(enthusiasm)

app = gr.Interface(
    fn=greet,
    inputs=[
        gr.Textbox(label="Your name"),
        gr.Slider(1, 5, value=1, label="Enthusiasm level"),
    ],
    outputs=gr.Textbox(label="Greeting"),
    title="My First Gradio App",
)
app.launch()
```

Run it: `python hello_ml.py` → opens at `http://localhost:7860`

### Step 3: Add a Plotly chart output

Gradio supports interactive Plotly charts natively with `gr.Plot()`:

```python
import gradio as gr
import plotly.express as px
import numpy as np

def sine_wave(frequency, amplitude):
    x = np.linspace(0, 4 * np.pi, 500)
    y = amplitude * np.sin(frequency * x)
    fig = px.line(x=x, y=y, title=f"Sine Wave: freq={frequency}, amp={amplitude}")
    return fig

app = gr.Interface(
    fn=sine_wave,
    inputs=[
        gr.Slider(0.5, 5.0, value=1.0, label="Frequency"),
        gr.Slider(0.5, 3.0, value=1.0, label="Amplitude"),
    ],
    outputs=gr.Plot(label="Wave"),
    title="Sine Wave Explorer",
    live=True,   # Updates on every slider change (no button needed)
)
app.launch()
```

### Step 4: Use `gr.Blocks` for more control

`gr.Interface` is simple but limited. `gr.Blocks` lets you control layout:

```python
import gradio as gr
import numpy as np
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

def explore_iris(n_components):
    data = load_iris()
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(data.data)

    if n_components == 2:
        fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1],
                         color=data.target.astype(str),
                         title="Iris PCA 2D")
    else:
        fig = px.scatter_3d(x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2],
                            color=data.target.astype(str),
                            title="Iris PCA 3D")
    return fig

with gr.Blocks(title="Iris Explorer") as app:
    gr.Markdown("# 🌸 Iris Dataset Explorer")

    with gr.Row():
        with gr.Column(scale=1):
            dims = gr.Radio([2, 3], value=2, label="PCA Dimensions")
            btn = gr.Button("▶ Run PCA", variant="primary")
        with gr.Column(scale=2):
            plot = gr.Plot(label="PCA Result")

    btn.click(fn=explore_iris, inputs=[dims], outputs=[plot])

app.launch()
```

### Step 5: Launch the full course app

```bash
cd lessons/02-ml-with-gradio/app
python app.py
```

Explore the first tab "📊 01 · Data Exploration". Notice how it uses the same `gr.Blocks` pattern with accordions for theory and an interactive demo.

## Key Concepts

| Concept | What It Does |
|---|---|
| `gr.Interface(fn, inputs, outputs)` | Simple wrapper — maps one function to inputs/outputs |
| `gr.Blocks()` | Full control over layout using `with` context managers |
| `gr.Row()` / `gr.Column()` | Side-by-side or stacked layout |
| `gr.Accordion()` | Collapsible section |
| `gr.Tab()` / `gr.Tabs()` | Tabbed interface |
| `component.click(fn, inputs, outputs)` | Wire button click to function |
| `live=True` | Update output on every input change |
| `gr.Plot()` | Renders Plotly figures interactively |

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: gradio` | Run `pip install gradio` |
| Port 7860 already in use | Add `server_port=7861` to `app.launch()` |
| App opens but shows error | Check terminal for Python traceback |
| Plot not rendering | Make sure function returns a Plotly `fig` object, not `None` |

---

Next: [Exercise 2 — Data Exploration →](./02-data-exploration.md)

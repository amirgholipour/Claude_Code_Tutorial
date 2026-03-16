"""
Plotly helper functions for architecture diagrams.
No Gradio imports — pure Python / Plotly utility.
"""

from __future__ import annotations

import math
from typing import Optional

import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# 1. Flow Diagram
# ---------------------------------------------------------------------------

def flow_diagram(
    nodes: list[dict],
    edges: list[dict],
    title: str,
) -> go.Figure:
    """
    Render an architecture flow diagram using Plotly scatter + line traces.

    Parameters
    ----------
    nodes : list of dicts with keys:
        - "id"    : str   — unique identifier
        - "label" : str   — display label
        - "x"     : float — horizontal position
        - "y"     : float — vertical position
        - "color" : str   — hex color (optional, defaults to primary blue)
    edges : list of dicts with keys:
        - "from"  : str   — source node id
        - "to"    : str   — target node id
        - "label" : str   — edge label (optional)
    title : str

    Returns
    -------
    go.Figure
    """
    pos = {n["id"]: (n["x"], n["y"]) for n in nodes}
    default_color = "#6366F1"

    # Build edge traces
    edge_traces = []
    annotation_list = []

    for edge in edges:
        x0, y0 = pos[edge["from"]]
        x1, y1 = pos[edge["to"]]
        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=2, color="#94A3B8"),
                hoverinfo="none",
                showlegend=False,
            )
        )
        # Midpoint label
        if edge.get("label"):
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            annotation_list.append(
                dict(
                    x=mx, y=my,
                    text=edge["label"],
                    showarrow=False,
                    font=dict(size=10, color="#64748B"),
                    bgcolor="rgba(255,255,255,0.8)",
                    borderpad=2,
                )
            )

    # Arrow heads (approximate via a short line ending)
    for edge in edges:
        x0, y0 = pos[edge["from"]]
        x1, y1 = pos[edge["to"]]
        # Place arrowhead annotation at destination
        annotation_list.append(
            dict(
                ax=x0, ay=y0,
                x=x1, y=y1,
                xref="x", yref="y",
                axref="x", ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.2,
                arrowwidth=2,
                arrowcolor="#94A3B8",
            )
        )

    # Node trace
    node_x = [n["x"] for n in nodes]
    node_y = [n["y"] for n in nodes]
    node_colors = [n.get("color", default_color) for n in nodes]
    node_labels = [n["label"] for n in nodes]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(
            size=40,
            color=node_colors,
            symbol="square",
            line=dict(width=2, color="white"),
        ),
        text=node_labels,
        textposition="middle center",
        textfont=dict(size=11, color="white", family="monospace"),
        hovertext=node_labels,
        hoverinfo="text",
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#1E293B")),
        xaxis=dict(visible=False, range=[min(node_x) - 1, max(node_x) + 1]),
        yaxis=dict(visible=False, range=[min(node_y) - 1, max(node_y) + 1]),
        plot_bgcolor="rgba(248,250,252,1)",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=50, b=20),
        annotations=annotation_list,
        height=420,
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Radar / Spider Chart
# ---------------------------------------------------------------------------

def radar_chart(
    categories: list[str],
    scores_dict: dict[str, list[float]],
    title: str,
) -> go.Figure:
    """
    Render a radar/spider chart comparing multiple options across categories.

    Parameters
    ----------
    categories  : list of axis labels, e.g. ["Scalability", "Consistency", "Speed"]
    scores_dict : {"PostgreSQL": [4, 2, 5, 3, 4], "MongoDB": [3, 5, 2, 4, 3]}
                  Values should be in range [0, 5]
    title       : str

    Returns
    -------
    go.Figure
    """
    colors = [
        "#6366F1", "#10B981", "#F59E0B", "#EF4444", "#3B82F6",
        "#8B5CF6", "#06B6D4", "#F97316",
    ]

    fig = go.Figure()
    for idx, (name, scores) in enumerate(scores_dict.items()):
        # Close the polygon
        closed_scores = list(scores) + [scores[0]]
        closed_cats = list(categories) + [categories[0]]
        color = colors[idx % len(colors)]
        fig.add_trace(
            go.Scatterpolar(
                r=closed_scores,
                theta=closed_cats,
                fill="toself",
                fillcolor=color.replace("#", "rgba(").rstrip(")") + ",0.15)",
                line=dict(color=color, width=2),
                name=name,
                marker=dict(size=6, color=color),
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5],
                tickvals=[1, 2, 3, 4, 5],
                tickfont=dict(size=9, color="#64748B"),
                gridcolor="#E2E8F0",
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color="#1E293B"),
                gridcolor="#E2E8F0",
            ),
            bgcolor="rgba(248,250,252,1)",
        ),
        title=dict(text=title, font=dict(size=16, color="#1E293B")),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.15,
            xanchor="center", x=0.5,
            font=dict(size=11),
        ),
        paper_bgcolor="white",
        margin=dict(l=60, r=60, t=60, b=60),
        height=420,
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Heatmap Comparison
# ---------------------------------------------------------------------------

def heatmap_comparison(
    rows: list[str],
    cols: list[str],
    values: list[list[float]],
    title: str,
) -> go.Figure:
    """
    Render a heatmap suitable for tool/platform comparison tables.

    Parameters
    ----------
    rows   : row labels (e.g., database names)
    cols   : column labels (e.g., criteria)
    values : 2D list [len(rows)][len(cols)], typically in [0, 5]
    title  : str

    Returns
    -------
    go.Figure
    """
    text_labels = [[f"{v:.1f}" for v in row] for row in values]

    fig = go.Figure(
        data=go.Heatmap(
            z=values,
            x=cols,
            y=rows,
            text=text_labels,
            texttemplate="%{text}",
            colorscale=[
                [0.0, "#FEF2F2"],
                [0.25, "#FEF9C3"],
                [0.5, "#DCFCE7"],
                [0.75, "#DBEAFE"],
                [1.0, "#6366F1"],
            ],
            zmin=0,
            zmax=5,
            showscale=True,
            colorbar=dict(
                title="Score",
                tickvals=[0, 1, 2, 3, 4, 5],
                thickness=12,
                len=0.8,
            ),
            hoverongaps=False,
            xgap=2,
            ygap=2,
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#1E293B")),
        xaxis=dict(
            tickangle=-30,
            tickfont=dict(size=11, color="#1E293B"),
            side="bottom",
        ),
        yaxis=dict(
            tickfont=dict(size=11, color="#1E293B"),
            autorange="reversed",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=120, r=40, t=60, b=80),
        height=max(300, len(rows) * 50 + 120),
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Network Graph (Agent Communication)
# ---------------------------------------------------------------------------

def network_graph(
    agents: list[dict],
    messages: list[dict],
    title: str,
    highlight_agent: Optional[str] = None,
) -> go.Figure:
    """
    Render an agent communication network diagram.

    Parameters
    ----------
    agents : list of dicts:
        - "id"    : str — unique identifier
        - "label" : str — display name
        - "role"  : str — "orchestrator" | "worker" | "tool" | "user"
    messages : list of dicts:
        - "from"  : str — sender agent id
        - "to"    : str — receiver agent id
        - "label" : str — message label / action
        - "step"  : int — sequence number
    title : str
    highlight_agent : str | None — agent id to mark as failed/disabled (red)

    Returns
    -------
    go.Figure
    """
    role_colors = {
        "orchestrator": "#6366F1",
        "worker": "#10B981",
        "tool": "#F59E0B",
        "user": "#3B82F6",
        "default": "#94A3B8",
    }

    n = len(agents)
    # Arrange agents in a circle
    positions = {}
    for i, agent in enumerate(agents):
        angle = 2 * math.pi * i / n - math.pi / 2
        positions[agent["id"]] = (math.cos(angle) * 2, math.sin(angle) * 2)

    # Edge traces with step labels
    edge_traces = []
    annotations = []

    for msg in messages:
        if msg["from"] not in positions or msg["to"] not in positions:
            continue
        x0, y0 = positions[msg["from"]]
        x1, y1 = positions[msg["to"]]
        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=1.5, color="#CBD5E1", dash="solid"),
                hoverinfo="none",
                showlegend=False,
            )
        )
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        label = f"[{msg.get('step', '?')}] {msg.get('label', '')}"
        annotations.append(
            dict(
                x=mx, y=my,
                text=label,
                showarrow=False,
                font=dict(size=9, color="#475569"),
                bgcolor="rgba(255,255,255,0.85)",
                borderpad=1,
            )
        )

    # Node trace
    node_x, node_y, node_colors, node_texts, hover_texts = [], [], [], [], []
    for agent in agents:
        x, y = positions[agent["id"]]
        node_x.append(x)
        node_y.append(y)

        if agent["id"] == highlight_agent:
            node_colors.append("#EF4444")  # red = failed
        else:
            node_colors.append(role_colors.get(agent.get("role", ""), role_colors["default"]))

        label = agent["label"]
        node_texts.append(label)
        hover_texts.append(f"{label}<br>Role: {agent.get('role', 'unknown')}")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(
            size=35,
            color=node_colors,
            symbol="circle",
            line=dict(width=2, color="white"),
        ),
        text=node_texts,
        textposition="top center",
        textfont=dict(size=10, color="#1E293B"),
        hovertext=hover_texts,
        hoverinfo="text",
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])

    # Role legend
    for role, color in role_colors.items():
        if role == "default":
            continue
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=10, color=color),
                name=role.capitalize(),
                showlegend=True,
            )
        )
    if highlight_agent:
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=10, color="#EF4444"),
                name="Failed Agent",
                showlegend=True,
            )
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#1E293B")),
        xaxis=dict(visible=False, range=[-3.5, 3.5]),
        yaxis=dict(visible=False, range=[-3.5, 3.5]),
        plot_bgcolor="rgba(248,250,252,1)",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.12,
            xanchor="center", x=0.5,
            font=dict(size=10),
        ),
        annotations=annotations,
        height=450,
    )
    return fig


# ---------------------------------------------------------------------------
# 5. Timeline Steps (ReAct Agent Trace)
# ---------------------------------------------------------------------------

def timeline_steps(
    steps: list[dict],
    title: str,
) -> go.Figure:
    """
    Render a horizontal timeline of ReAct agent steps.

    Parameters
    ----------
    steps : list of dicts:
        - "step"        : int  — step number
        - "agent"       : str  — agent name
        - "type"        : str  — "Thought" | "Action" | "Observation" | "Final"
        - "content"     : str  — brief content/summary
        - "duration_ms" : int  — duration in milliseconds
    title : str

    Returns
    -------
    go.Figure
    """
    type_colors = {
        "Thought":     "#6366F1",
        "Action":      "#F59E0B",
        "Observation": "#10B981",
        "Final":       "#3B82F6",
        "Error":       "#EF4444",
    }
    type_symbols = {
        "Thought":     "💭",
        "Action":      "⚡",
        "Observation": "👁",
        "Final":       "✅",
        "Error":       "❌",
    }

    if not steps:
        fig = go.Figure()
        fig.update_layout(title=title, height=200)
        return fig

    n = len(steps)
    x_positions = list(range(n))

    # Horizontal timeline line
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_positions,
            y=[0] * n,
            mode="lines",
            line=dict(color="#E2E8F0", width=3),
            hoverinfo="none",
            showlegend=False,
        )
    )

    # Group by type for legend
    added_types = set()
    for i, step in enumerate(steps):
        stype = step.get("type", "Thought")
        color = type_colors.get(stype, "#94A3B8")
        symbol = type_symbols.get(stype, "•")
        duration = step.get("duration_ms", 0)
        content = step.get("content", "")
        # Truncate long content for hover
        hover = (
            f"Step {step.get('step', i+1)}: {stype}<br>"
            f"Agent: {step.get('agent', 'Unknown')}<br>"
            f"Duration: {duration} ms<br>"
            f"Content: {content[:120]}{'...' if len(content) > 120 else ''}"
        )

        show_legend = stype not in added_types
        added_types.add(stype)

        fig.add_trace(
            go.Scatter(
                x=[i],
                y=[0],
                mode="markers+text",
                marker=dict(
                    size=22,
                    color=color,
                    symbol="circle",
                    line=dict(width=2, color="white"),
                ),
                text=[f"{symbol}"],
                textposition="middle center",
                textfont=dict(size=12),
                hovertext=[hover],
                hoverinfo="text",
                name=stype,
                showlegend=show_legend,
                legendgroup=stype,
            )
        )

        # Step label below
        fig.add_annotation(
            x=i, y=-0.4,
            text=f"<b>{stype}</b><br><span style='font-size:9px'>{content[:30]}{'…' if len(content)>30 else ''}</span>",
            showarrow=False,
            font=dict(size=9, color="#475569"),
            align="center",
        )

        # Duration label above
        if duration:
            fig.add_annotation(
                x=i, y=0.3,
                text=f"{duration}ms",
                showarrow=False,
                font=dict(size=8, color="#94A3B8"),
                align="center",
            )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#1E293B")),
        xaxis=dict(
            visible=False,
            range=[-0.8, n - 0.2],
        ),
        yaxis=dict(
            visible=False,
            range=[-1.2, 0.8],
        ),
        plot_bgcolor="rgba(248,250,252,1)",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.12,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        height=260,
    )
    return fig

# Claude Code Tutorial

A hands-on tutorial series for learning real-world development workflows using Claude Code.

## What This Is

Each lesson teaches a practical workflow through guided exercises and custom slash commands. You learn by doing — Claude executes the commands alongside you.

## Lessons

| # | Topic | Status |
|---|---|---|
| 01 | [GitHub Repo Management](./lessons/01-github-repo-management/README.md) | Ready |
| 02 | [Machine Learning & Deep Learning with Gradio](./lessons/02-ml-with-gradio/README.md) | Ready |

## How It Works

1. Open Claude Code in this project directory
2. Open the lesson you want (`lessons/01-.../README.md`)
3. Follow the numbered exercises
4. Use the provided slash commands — Claude handles the heavy lifting

## Prerequisites

- Git installed (`git --version`)
- GitHub CLI installed and authenticated (`gh auth status`)
- Python 3.9+ installed (`python --version`)
- Claude Code installed

## Custom Slash Commands

All commands are **project-scoped** — available when Claude Code is open in this directory.

**Lesson 1 — GitHub:**
```
/github-create-repo   — Create a new GitHub repo from a local directory
/github-branch        — Create, list, or switch branches
/github-pr            — Create, list, or merge pull requests
/github-issue         — Create, list, or close GitHub issues
/github-release       — Tag a version and publish a GitHub release
```

**Lesson 2 — Machine Learning:**
```
/ml-setup             — Install all Python dependencies for the ML app
/ml-run-app           — Launch the ML & Deep Learning Gradio app
```

These are **project-scoped** — they only appear when Claude Code is open in this directory.

## Getting Started

```bash
# 1. Open Claude Code in this directory
# 2. Start Lesson 1
```

Open [Lesson 1 →](./lessons/01-github-repo-management/README.md)

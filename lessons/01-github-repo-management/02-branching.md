# Exercise 2: Branching

## Goal

Create a feature branch, make a change, and push it to GitHub.

## What You'll Learn

- Why branches matter (isolating work from `main`)
- How to create and switch branches
- How to push a branch to GitHub

## Background

The golden rule of Git: **never commit directly to `main`**.

Branches let multiple people (or your future self) work on different features simultaneously without breaking the main codebase. A typical workflow:

```
main ──────────────────────────────→
         └── feature/add-readme ──→ (PR) → merged back to main
```

## Steps

### Step 1: Check your current branch

In Claude Code, run:
```
/github-branch
```

Claude will list all branches and highlight your current one (`main`).

### Step 2: Create a feature branch

Ask Claude:
```
/github-branch create a branch called feature/add-content
```

Claude will:
1. Run `git checkout -b feature/add-content`
2. Push it to GitHub: `git push -u origin feature/add-content`

You are now on the new branch.

### Step 3: Make a change

Create or edit a file. For example, tell Claude:
```
Create a file called notes.txt with a short description of what this repo is for
```

Then ask Claude to commit it:
```
/commit
```

### Step 4: Verify on GitHub

Visit your GitHub repo and look for the branch dropdown — you should see `feature/add-content` listed.

Or use the command line:
```bash
git branch -a
```

## Key Concepts

- **`git checkout -b <name>`** — creates a new branch AND switches to it
- **`git push -u origin <name>`** — pushes the branch to GitHub and sets up tracking
- **`-u` flag** — sets the upstream, so future `git push` works without specifying the remote
- **Branch naming conventions**: `feature/`, `fix/`, `docs/`, `chore/` prefixes help organize work

## Branch Naming Guide

| Prefix | Use for |
|---|---|
| `feature/` | New features or additions |
| `fix/` | Bug fixes |
| `docs/` | Documentation changes |
| `chore/` | Maintenance, config, tooling |
| `release/` | Release preparation |

## Troubleshooting

| Problem | Fix |
|---|---|
| `fatal: not a git repository` | Run `/github-create-repo` first (Exercise 1) |
| `error: src refspec does not match any` | Make sure you have at least one commit first |
| Already on a branch and want to go back | `/github-branch switch to main` |

---

Next: [Exercise 3 — Pull Requests →](./03-pull-requests.md)

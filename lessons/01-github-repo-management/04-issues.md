# Exercise 4: GitHub Issues

## Goal

Create a GitHub issue to track a task, then close it by linking it to a pull request.

## What You'll Learn

- How to create and manage GitHub Issues
- How to use issue numbers to track work
- How to automatically close issues via PR descriptions

## Background

GitHub Issues are the primary way to track bugs, features, and tasks. They integrate tightly with branches and PRs:

```
Issue #5: "Add a .gitignore file"
    └── Branch: feature/add-gitignore
        └── PR: "Add .gitignore" → body: "Closes #5"
            └── Merge → Issue #5 automatically closed
```

## Steps

### Step 1: Create an issue

In Claude Code, run:
```
/github-issue
```

Claude will ask for:
- **Title**: e.g., `Add a .gitignore file`
- **Body**: e.g., `The repo needs a .gitignore to avoid committing unnecessary files`
- **Label**: e.g., `enhancement`

Claude will create the issue and show you the issue number (e.g., `#3`).

### Step 2: Work on the issue

Create a feature branch for the issue:
```
/github-branch create feature/add-gitignore
```

Then ask Claude to create the file:
```
Create a .gitignore for a Python project
```

Commit the change:
```
/commit
```

### Step 3: Create a PR that closes the issue

Run:
```
/github-pr
```

When Claude drafts the PR body, make sure it includes:
```
Closes #3
```
(Replace `3` with your actual issue number.)

### Step 4: Merge and verify

Merge the PR. Then check:
```
/github-issue
```

The issue should now show as **closed** automatically.

## Issue Labels

| Label | Meaning |
|---|---|
| `bug` | Something isn't working |
| `enhancement` | New feature or request |
| `documentation` | Improvements or additions to docs |
| `good first issue` | Good for newcomers |
| `help wanted` | Extra attention is needed |
| `question` | Further information is requested |

## Key Concepts

- **Issue number** (`#3`) — unique identifier you reference in commits and PRs
- **`Closes #N`** — magic keyword: closing a PR with this in the body auto-closes the issue
- **Other keywords**: `Fixes #N`, `Resolves #N` all work the same way
- **Issue templates** — you can create `.github/ISSUE_TEMPLATE/` for structured issue forms

## Troubleshooting

| Problem | Fix |
|---|---|
| Issue not closing after PR merge | Make sure `Closes #N` is in the PR body, not a comment |
| Can't create issues | Repo must exist on GitHub — run Exercise 1 first |
| Issue not found | Use `/github-issue` to list and find the correct number |

---

Next: [Exercise 5 — Releases →](./05-releases.md)

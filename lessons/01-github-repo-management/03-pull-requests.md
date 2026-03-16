# Exercise 3: Pull Requests

## Goal

Open a pull request for your feature branch and merge it into `main`.

## What You'll Learn

- How to create a pull request (PR) with a proper description
- How PRs are used for code review before merging
- How to merge a PR

## Background

A **Pull Request** is a formal request to merge changes from one branch into another. Even when working solo, PRs are a good habit because they:
- Create a record of what changed and why
- Force you to write a description of your work
- Allow review before changes hit `main`

## Prerequisites

Complete Exercise 2 first — you need a feature branch with commits.

## Steps

### Step 1: Make sure you have commits on your feature branch

Run:
```bash
git log main..HEAD --oneline
```

You should see at least one commit listed. If not, go back to Exercise 2 and make a change.

### Step 2: Create the PR

In Claude Code, run:
```
/github-pr
```

Claude will:
1. Detect your current branch and commits
2. Draft a PR title from your commit messages
3. Draft a PR body summarizing the changes
4. Show you the draft and ask for confirmation
5. Run `gh pr create` and return the PR URL

### Step 3: Review the PR on GitHub

Open the PR URL in your browser. You'll see:
- Title and description
- Files changed tab (diff view)
- Commits tab

### Step 4: Merge the PR

Ask Claude:
```
/github-pr merge
```

Claude will run `gh pr merge --squash`, merging your branch into `main`.

### Step 5: Clean up

After merging, the feature branch is no longer needed. Ask Claude:
```
/github-branch delete feature/add-content
```

Then switch back to main and pull the latest:
```bash
git checkout main
git pull
```

## What a Good PR Description Looks Like

```markdown
## Summary
- Added notes.txt with project description
- Updated README with usage instructions

## Changes
- notes.txt (new file)
- README.md (updated)

Closes #1
```

## Key Concepts

- **`gh pr create`** — creates a PR on GitHub from the current branch
- **Squash merge** — combines all branch commits into a single commit on `main` (cleaner history)
- **Draft PR** — a PR marked as not ready for review (use `--draft` flag)
- **`Closes #N`** — automatically closes issue #N when the PR is merged

## Troubleshooting

| Problem | Fix |
|---|---|
| `No commits between main and HEAD` | Make sure you have commits on your feature branch |
| `already exists` PR error | A PR for this branch already exists — use `/github-pr list` |
| Can't merge (conflicts) | Resolve conflicts locally, then push again |

---

Next: [Exercise 4 — Issues →](./04-issues.md)

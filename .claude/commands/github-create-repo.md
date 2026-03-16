---
allowed-tools: Bash(git init:*), Bash(git remote add:*), Bash(git remote -v:*), Bash(git status:*), Bash(git add:*), Bash(git commit:*), Bash(git push:*), Bash(git branch:*), Bash(gh repo create:*), Bash(gh auth status:*)
description: Initialize local git and create a new GitHub repository
---

## Context

- Working directory: !`pwd`
- Git status: !`git status 2>/dev/null || echo "Not a git repo yet"`
- Existing remotes: !`git remote -v 2>/dev/null || echo "No remotes"`
- GitHub auth status: !`gh auth status 2>&1 | head -3`

## Your task

Create a new GitHub repository from the current directory.

**Steps to follow:**

1. If the user did not specify a repo name, ask: "What should the repo be named? (default: current directory name)" and visibility: "Public or private?"
2. Check if `gh auth status` shows authenticated. If not, tell the user to run `gh auth login` and stop.
3. If not already a git repo, run `git init` and create an initial commit:
   - `git add .` (or create a README.md if directory is empty)
   - `git commit -m "Initial commit"`
4. Run `gh repo create <name> --<public|private> --source=. --remote=origin --push`
5. Confirm success by showing the repo URL from the output.
6. Run `git remote -v` to verify the remote was added.

**Show the user:**
- The full GitHub URL of the created repo
- The default branch name
- Next suggested step: "You're ready! Try `/github-branch` to create your first feature branch."

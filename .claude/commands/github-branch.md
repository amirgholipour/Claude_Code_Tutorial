---
allowed-tools: Bash(git branch:*), Bash(git checkout:*), Bash(git switch:*), Bash(git push:*), Bash(git fetch:*), Bash(git status:*)
description: Create, list, switch, or delete git branches
---

## Context

- Current branch: !`git branch --show-current 2>/dev/null || echo "Not a git repo"`
- All branches: !`git branch -a 2>/dev/null || echo "No branches"`
- Git status: !`git status --short 2>/dev/null`

## Your task

Manage git branches based on the user's request. Determine the intent:

**If the user wants to CREATE a new branch:**
1. Ask for the branch name if not provided (suggest format: `feature/description` or `fix/description`)
2. Run `git checkout -b <branch-name>` (creates and switches to it)
3. Push with tracking: `git push -u origin <branch-name>`
4. Confirm: "Created and switched to branch `<name>`. Tracking origin/<name>."

**If the user wants to LIST branches:**
1. Show all local branches with `git branch`
2. Show remote branches with `git branch -r`
3. Highlight the current branch

**If the user wants to SWITCH to an existing branch:**
1. Run `git checkout <branch-name>` or `git switch <branch-name>`
2. Confirm the switch succeeded

**If the user wants to DELETE a branch:**
1. Confirm the branch name
2. Run `git branch -d <branch-name>` (safe delete — only if merged)
3. To delete from remote: `git push origin --delete <branch-name>`
4. Warn if the branch has unmerged changes

**Always show** the current branch after any operation.

---
allowed-tools: Bash(gh issue create:*), Bash(gh issue list:*), Bash(gh issue view:*), Bash(gh issue close:*), Bash(gh issue edit:*)
description: Create, list, view, or close GitHub issues
---

## Context

- Open issues: !`gh issue list --limit 10 2>/dev/null || echo "Could not fetch issues (is this a GitHub repo?)"`
- Current branch: !`git branch --show-current 2>/dev/null`

## Your task

Manage GitHub issues based on the user's intent:

**If the user wants to CREATE an issue:**
1. Ask for:
   - **Title**: Brief description of the issue
   - **Body**: Steps to reproduce / feature description (optional)
   - **Labels**: e.g., `bug`, `enhancement`, `documentation` (optional)
2. Run: `gh issue create --title "<title>" --body "<body>"`
3. Show the issue number and URL.
4. Tip: "Reference this issue in your PR description with `Closes #<number>`."

**If the user wants to LIST issues:**
1. Run `gh issue list` to show all open issues.
2. Show: number, title, labels, and date created.
3. If they want closed issues: `gh issue list --state closed`

**If the user wants to VIEW an issue:**
1. Ask for issue number if not provided.
2. Run `gh issue view <number>` and display full details.

**If the user wants to CLOSE an issue:**
1. Ask for issue number if not provided.
2. Ask for an optional closing comment.
3. Run `gh issue close <number> --comment "<reason>"`
4. Confirm: "Issue #<number> closed."

**Note:** Issues are automatically closed when a PR with `Closes #<number>` in the description is merged.

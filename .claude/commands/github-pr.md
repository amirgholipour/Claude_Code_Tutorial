---
allowed-tools: Bash(gh pr create:*), Bash(gh pr list:*), Bash(gh pr view:*), Bash(gh pr merge:*), Bash(git log:*), Bash(git diff:*), Bash(git branch:*)
description: Create, list, view, or merge a GitHub pull request
---

## Context

- Current branch: !`git branch --show-current 2>/dev/null`
- Commits ahead of main: !`git log main..HEAD --oneline 2>/dev/null || git log origin/main..HEAD --oneline 2>/dev/null || echo "Could not compare with main"`
- Open PRs: !`gh pr list 2>/dev/null || echo "Could not fetch PRs"`
- Diff summary: !`git diff main...HEAD --stat 2>/dev/null | tail -5`

## Your task

Manage pull requests based on the user's intent:

**If the user wants to CREATE a PR:**
1. Check that the current branch is not `main`. If it is, stop and say: "Switch to a feature branch first with `/github-branch`."
2. Use the commit history to draft a PR title (first commit message or summarize all commits).
3. Draft the PR body using this template:
   ```
   ## Summary
   - <bullet points from commit messages>

   ## Changes
   - <what files/areas changed>

   Closes #<issue-number if mentioned>
   ```
4. Ask the user to confirm or edit the title and body before creating.
5. Run: `gh pr create --title "<title>" --body "<body>"`
6. Return the PR URL.

**If the user wants to LIST PRs:**
1. Run `gh pr list` and display all open PRs with number, title, author, and branch.

**If the user wants to VIEW a PR:**
1. Ask for PR number if not provided.
2. Run `gh pr view <number>` and display details.

**If the user wants to MERGE a PR:**
1. Ask for PR number if not provided.
2. Run `gh pr merge <number> --squash` (default to squash merge).
3. Confirm merge success and show the merged commit.

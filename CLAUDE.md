# Claude Code Tutorial — GitHub Repo Management

## Project Purpose

This is a hands-on tutorial for learning GitHub repository management using Claude Code. You will learn how to create repositories, manage branches, open pull requests, track issues, and publish releases — all guided by Claude.

## Prerequisites

Before starting, ensure you have:
- **Git** installed and configured (`git config --global user.name` and `user.email`)
- **GitHub CLI** installed and authenticated: `gh auth login`
- A **GitHub account**
- **Claude Code** installed

Verify setup:
```bash
git --version
gh auth status
```

## Project Structure

```
Claude_Code_Tutorial/
├── CLAUDE.md                          # This file — Claude's instructions
├── README.md                          # Project overview
├── lessons/
│   └── 01-github-repo-management/    # Lesson 1 content
│       ├── README.md                  # Lesson overview & objectives
│       ├── 01-create-repo.md          # Exercise: Create a repo
│       ├── 02-branching.md            # Exercise: Branching
│       ├── 03-pull-requests.md        # Exercise: Pull requests
│       ├── 04-issues.md               # Exercise: Issues
│       └── 05-releases.md             # Exercise: Releases
└── .claude/
    └── commands/                      # Custom slash commands
        ├── github-create-repo.md      # /github-create-repo
        ├── github-branch.md           # /github-branch
        ├── github-pr.md               # /github-pr
        ├── github-issue.md            # /github-issue
        └── github-release.md          # /github-release
```

## Available Slash Commands

These commands are active in this project. Use them to perform GitHub actions:

| Command | What it does |
|---|---|
| `/github-create-repo` | Initialize git and create a new GitHub repo |
| `/github-branch` | Create, list, or switch branches |
| `/github-pr` | Create a pull request with auto-drafted description |
| `/github-issue` | Create, list, or close GitHub issues |
| `/github-release` | Tag a version and publish a GitHub release |

## Workflow Rules

1. **Always check git status first** — before any git operation, verify the current state
2. **Never commit directly to main** — use feature branches for all changes
3. **PRs require a description** — auto-draft from commit history, then review before submitting
4. **Link issues to PRs** — use `Closes #<issue-number>` in PR descriptions
5. **Verify before done** — confirm the GitHub action succeeded (check the URL returned)

## How to Use This Tutorial

1. Open the lesson: `lessons/01-github-repo-management/README.md`
2. Follow the numbered exercise files in order
3. Use the slash commands listed above to execute each step
4. Each exercise file explains what to do and which command to run

## Claude's Behavior in This Project

- Always confirm git/gh CLI is available before running commands
- When creating a repo, ask for the repo name and visibility (public/private) if not specified
- Show the GitHub URL after every repo operation
- When something fails, show the error and suggest the fix
- Prefer `gh` CLI over raw `curl` for all GitHub API calls

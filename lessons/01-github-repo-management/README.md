# Lesson 1: GitHub Repository Management

## Overview

In this lesson you will create a GitHub repository from scratch and go through the full lifecycle of repo management: branching, pull requests, issues, and releases — all using Claude Code slash commands.

By the end of this lesson, you will have:
- A live GitHub repository you created yourself
- Experience with the core GitHub workflow used by professional teams
- A set of reusable slash commands you can use in any project

## Learning Objectives

1. Create a new GitHub repository from a local directory
2. Work with feature branches (create, switch, push)
3. Open and merge pull requests with proper descriptions
4. Track work with GitHub Issues
5. Tag versions and publish releases

## Prerequisites

- [ ] Git installed: `git --version`
- [ ] GitHub CLI installed and authenticated: `gh auth status`
- [ ] Claude Code running in this project directory

## Exercises

Work through these in order:

| # | File | What you'll do |
|---|---|---|
| 1 | [01-create-repo.md](./01-create-repo.md) | Create a GitHub repo from scratch |
| 2 | [02-branching.md](./02-branching.md) | Create and manage feature branches |
| 3 | [03-pull-requests.md](./03-pull-requests.md) | Open and merge a pull request |
| 4 | [04-issues.md](./04-issues.md) | Create and close GitHub issues |
| 5 | [05-releases.md](./05-releases.md) | Tag a version and publish a release |

## Available Commands

All commands below are active when Claude Code is open in this project:

```
/github-create-repo   — Create a new GitHub repo
/github-branch        — Manage branches
/github-pr            — Create/list/merge pull requests
/github-issue         — Create/list/close issues
/github-release       — Tag versions and publish releases
```

## Time Estimate

~45–60 minutes to complete all exercises.

---

Ready? Start with [Exercise 1 →](./01-create-repo.md)

# Exercise 1: Create a GitHub Repository

## Goal

Create a brand new GitHub repository from scratch using Claude Code.

## What You'll Learn

- How to initialize a local git repository
- How to create a matching GitHub repository using `gh repo create`
- How to link local ↔ remote and push your first commit

## Background

When you start a new project, you typically need to:
1. Initialize a local git repo (`git init`)
2. Create the repo on GitHub
3. Connect them with a remote (`git remote add origin`)
4. Push your first commit

The `/github-create-repo` command handles all of this in one step.

## Steps

### Step 1: Create a practice directory

Open a terminal and create a new project directory to practice with:

```bash
mkdir my-first-repo
cd my-first-repo
```

Open Claude Code in that directory.

### Step 2: Run the command

In Claude Code, type:

```
/github-create-repo
```

Claude will ask you:
- **Repo name**: Enter something like `my-first-repo` or any name you like
- **Visibility**: `public` or `private` (choose private for practice)

### Step 3: Observe what happens

Watch Claude execute these steps automatically:
1. `git init` — initialize local repo
2. Create a `README.md` if the directory is empty
3. `git add .` and `git commit -m "Initial commit"`
4. `gh repo create` — create the repo on GitHub
5. `git push -u origin main` — push to GitHub

### Step 4: Verify

After the command completes, you should see:
- A GitHub URL like `https://github.com/<your-username>/my-first-repo`
- `git remote -v` shows `origin` pointing to that URL

**Check it yourself:**
```bash
git remote -v
git log --oneline
```

Visit the GitHub URL in your browser to confirm the repo exists.

## What to Expect

```
Created repository yourname/my-first-repo on GitHub
  https://github.com/yourname/my-first-repo
Branch 'main' set up to track remote branch 'main' from 'origin'.
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `gh: command not found` | Install GitHub CLI: https://cli.github.com |
| `not logged in` | Run `gh auth login` and follow prompts |
| `Repository name already exists` | Choose a different name |
| `git: command not found` | Install Git from https://git-scm.com |

## Key Concepts

- **`git init`** — turns a directory into a git repository
- **`gh repo create`** — GitHub CLI command to create a repo on GitHub
- **`--source=.`** — tells `gh` to use the current directory as the source
- **`--push`** — automatically pushes the initial commit

---

Next: [Exercise 2 — Branching →](./02-branching.md)

# Exercise 5: Releases

## Goal

Tag a version and publish a GitHub Release with auto-generated release notes.

## What You'll Learn

- How semantic versioning (`v1.0.0`) works
- How to create git tags
- How to publish a GitHub Release with a changelog

## Background

A **Release** is a snapshot of your code at a specific point in time, marked with a version tag. This is how you communicate "this is a stable version" to users.

```
v0.1.0 — initial release
v0.1.1 — bug fix
v0.2.0 — new feature added
v1.0.0 — stable, production-ready
```

## Semantic Versioning (SemVer)

The format is `vMAJOR.MINOR.PATCH`:

| Part | When to bump |
|---|---|
| MAJOR | Breaking changes — old code won't work |
| MINOR | New features — backwards compatible |
| PATCH | Bug fixes — backwards compatible |

## Steps

### Step 1: Make sure you have commits on `main`

Verify you've completed Exercises 1–4 and have merged at least one PR. Check:
```bash
git log --oneline
```

### Step 2: Create a release

In Claude Code, run:
```
/github-release
```

Claude will:
1. Look at your git tags — if none exist, suggest `v0.1.0`
2. Show you recent commits to use as release notes
3. Ask you to confirm the version
4. Create the git tag and push it
5. Create the GitHub Release with generated notes
6. Return the release URL

### Step 3: View the release on GitHub

Visit your repo on GitHub. You'll see a **Releases** section in the right sidebar. Click it to see your release with:
- Version tag
- Release notes
- Date

### Step 4: Create a second release (optional)

Make another small change, commit it, merge it to `main`, then run `/github-release` again. This time it will suggest `v0.1.1` and show only the commits since `v0.1.0`.

## What a Good Release Looks Like

```
## v0.1.0 — 2026-03-16

### What's Changed
- feat: add notes.txt with project description
- feat: add .gitignore for Python
- docs: add README with usage instructions

### First release
Initial release of the project.
```

## Key Concepts

- **`git tag v0.1.0`** — creates a lightweight tag at the current commit
- **`git push origin v0.1.0`** — pushes the tag to GitHub (tags don't push with normal `git push`)
- **`gh release create`** — creates the release on GitHub using the tag
- **`--generate-notes`** — lets GitHub auto-generate release notes from PR titles

## Troubleshooting

| Problem | Fix |
|---|---|
| `tag already exists` | Use a different version number or `git tag -d v0.1.0` to delete locally |
| `Release not showing on GitHub` | Make sure the tag was pushed: `git push origin --tags` |
| Commits not showing in release notes | Make sure you're looking at commits since the previous tag |

## Congratulations!

You've completed Lesson 1. You now know how to:
- Create a GitHub repo from scratch
- Work with feature branches
- Open and merge pull requests
- Track work with issues
- Publish versioned releases

These are the core skills used in every professional GitHub workflow.

---

[← Back to Lesson 1 Overview](./README.md)

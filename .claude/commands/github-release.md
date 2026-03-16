---
allowed-tools: Bash(gh release create:*), Bash(gh release list:*), Bash(gh release view:*), Bash(git tag:*), Bash(git log:*), Bash(git push:*)
description: Tag a version and create a GitHub release with changelog
---

## Context

- Latest tag: !`git describe --tags --abbrev=0 2>/dev/null || echo "No tags yet"`
- Recent commits since last tag: !`git log $(git describe --tags --abbrev=0 2>/dev/null || echo "HEAD~10")..HEAD --oneline 2>/dev/null | head -20`
- All existing tags: !`git tag --sort=-version:refname 2>/dev/null | head -10`
- Existing releases: !`gh release list --limit 5 2>/dev/null || echo "No releases yet"`

## Your task

Create a GitHub release based on the user's request:

**If the user wants to CREATE a release:**
1. Suggest a version number based on the latest tag:
   - No previous tags → suggest `v0.1.0`
   - Has tags → suggest bumping patch (e.g., `v1.0.0` → `v1.0.1`) or ask if it's a minor/major bump
2. Ask the user to confirm the version: "Create release `<suggested-version>`? (or specify your own)"
3. Generate release notes from commits since the last tag, grouped by type:
   ```
   ## What's Changed
   - feat: <commit message>
   - fix: <commit message>
   - docs: <commit message>
   ```
4. Create the tag and push it: `git tag <version> && git push origin <version>`
5. Create the release: `gh release create <version> --title "<version>" --notes "<notes>"`
6. Return the release URL.

**If the user wants to LIST releases:**
1. Run `gh release list` and display all releases.

**If the user wants to VIEW a release:**
1. Ask for tag/version if not provided.
2. Run `gh release view <tag>` and display details.

**Versioning guide:**
- `v0.x.x` — early development / not stable
- Patch bump (`x.x.1`) — bug fixes, small tweaks
- Minor bump (`x.1.x`) — new features, backwards compatible
- Major bump (`1.x.x`) — breaking changes

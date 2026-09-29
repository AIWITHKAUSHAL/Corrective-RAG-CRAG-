# Publishing and submission

The assignment asks for a GitHub repository link and a YouTube video link. Publish the finished code and record your explanation before submitting those two URLs.

## 1. Validate locally

```bash
uv sync --frozen
uv run pytest -q
uv run ruff check .
uv run python scripts/build_visualizer.py
uv run python -m crag "My bot finds junk" --mode demo
# Once EURI_API_KEY is configured:
uv run python -m crag "What is CRAG?" --mode live --output artifacts/live-check.json
```

## 2. Publish to your GitHub account

If GitHub CLI authentication is unavailable, run `gh auth login` and sign into the intended account. Choose your own repository owner/name; the following commands create a new repository from this folder:

```bash
git init -b main
git add .
git diff --cached --stat
# Verify .env is not staged before committing.
git status --short
git commit -m "Build LangGraph Corrective RAG lab with EURI and visualizer"
gh repo create Corrective-RAG --public --source=. --remote=origin --push
```

If the folder is already a Git repository, skip `git init`. If an `origin` remote already exists, use its normal `git push` workflow instead of creating another repository. Review visibility and owner before publishing.

Never force-add `.env`, local keys, or credentials. The supplied `.gitignore` excludes those files. Live traces contain your question and retrieved text; inspect them before deliberately sharing any trace.

## 3. Optionally enable the architecture explorer on GitHub Pages

In the repository's **Settings → Pages**, select deployment from the `main` branch and `/docs` folder. After Pages reports a successful deployment, use:

```text
https://YOUR_USERNAME.github.io/Corrective-RAG/architecture_visualizer.html
```

This is a URL pattern, not an already published link. GitHub Pages serves the standalone explorer; run the FastAPI playground locally for live model requests.

## 4. Record and upload your video

Use [VIDEO_GUIDE.md](VIDEO_GUIDE.md). Cover every LangGraph node, both conditional decisions, the correction loop, and final outputs. Include a successful live EURI example if you have provider access. Upload through your YouTube account, select visibility that your reviewer can access, and confirm the video plays.

## 5. Submit the two real links

```text
GitHub Repository: [your published GitHub URL]
YouTube Explanation: [your uploaded YouTube video URL]
```

Before submitting, open both links in a signed-out browser to verify reviewer access. The due-date string supplied with the assignment was `06/10/2026, 20:26:00`; confirm its date format and timezone in your course portal.

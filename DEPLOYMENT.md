# Deploying the Application for Free — Step by Step

The app is a single FastAPI process (`uvicorn qa_assistant.api:app`) that serves both the web UI and the JSON API. It has no database and needs about 150 MB of RAM, so any free Python or Docker host can run it.

| Option | Cost / card needed | Good for | Catch |
|---|---|---|---|
| **A. Render** (recommended) | Free, no card | Easiest: deploys straight from GitHub, HTTPS URL included | Sleeps after 15 min without visitors; the first request after that takes ~1 min |
| **B. Hugging Face Spaces** | Free, no card | Stays awake much longer (sleeps after ~48 h idle) | Code is pushed to a second git remote (a few extra commands); Space is public by default |
| **C. Your own Linux server** (e.g. Oracle Cloud Always Free VM) | Free, card needed for sign-up | Always on, full control | You maintain the server yourself |

Free-tier limits change from time to time; check the host's pricing page if something below no longer matches.

Files used for deployment (already in the repository):
- `render.yaml` — Render Blueprint (option A)
- `Dockerfile`, `.dockerignore` — container image (option B, and any Docker host)

---

## Step 1 — Prepare (all options)

### 1.1 Get a Gemini API key
Create one at <https://aistudio.google.com/apikey>. Without a key the deployed app still works, in offline (extractive) mode.

### 1.2 Make sure the key is NOT in git
The key belongs only in the host's secret settings, never in the repository.
```bash
git ls-files .env        # must print nothing
git check-ignore .env    # must print ".env"
```

### 1.3 Run the tests
```bash
source .venv/bin/activate
python -m pytest -q
```

### 1.4 Commit and push to GitHub
```bash
git add -A
git status                # check .env is NOT listed
git commit -m "Add deployment configuration"
git push origin main
```
The repository is `https://github.com/ranakaushik/context-aware-qa`. Render can deploy from public **or** private GitHub repositories.

---

## Option A — Render (recommended)

### A.1 Create an account
1. Open <https://render.com> and click **Get Started**.
2. Sign up with **GitHub** (this also lets Render see your repositories).

### A.2 Create the service from the Blueprint
1. In the Render dashboard click **New +** → **Blueprint**.
2. Connect GitHub if asked, and give Render access to the `context-aware-qa` repository.
3. Select the repository. Render reads `render.yaml` and shows one service: `context-aware-qa` (Free plan).
4. Render asks for **GEMINI_API_KEY** — paste your key.
5. Click **Apply** / **Deploy Blueprint**.

What `render.yaml` sets up:

| Setting | Value |
|---|---|
| Build command | `pip install -r requirements.txt` |
| Start command | `uvicorn qa_assistant.api:app --host 0.0.0.0 --port $PORT` |
| Health check | `/api/health` |
| Python | 3.12.7 |
| Environment | `GEMINI_API_KEY` (you enter it), `GEMINI_MODEL=gemini-flash-latest`, `QA_MODE=auto` |

> **Without the Blueprint** (manual setup): **New +** → **Web Service** → pick the repository → Runtime **Python 3**, Build Command and Start Command as in the table, Instance Type **Free**. Under **Environment** add `PYTHON_VERSION=3.12.7`, `GEMINI_API_KEY=<your key>` and any other variables from the table.

### A.3 Wait for the build
Open the service and watch **Logs**. A successful deploy ends with lines like:
```
Uvicorn running on http://0.0.0.0:10000
Your service is live 🎉
```
This takes 2–4 minutes the first time.

### A.4 Open and verify the app
Your URL is shown at the top of the service page, e.g. `https://context-aware-qa.onrender.com` (Render adds a suffix if the name is taken).

1. `https://<your-url>/api/health` should return `"status":"ok"` and `"mode":"gemini"` (or `"offline"` if no key was given).
2. `https://<your-url>/` opens the web UI. Ask: *Can employees carry forward more than 10 days of leave?*

### A.5 Updating the app
Every `git push origin main` triggers a new deploy automatically (`autoDeploy: true`). Change environment variables under the service's **Environment** tab; saving them redeploys.

### A.6 About sleeping (free plan)
A free Render service sleeps after 15 minutes with no traffic; the next visitor waits about a minute while it wakes. That is fine for demos. To keep it awake, a free monitor such as <https://cron-job.org> or UptimeRobot can request `https://<your-url>/api/health` every 10 minutes. One service running all month fits within Render's 750 free instance hours.

---

## Option B — Hugging Face Spaces (Docker)

### B.1 Create the Space
1. Sign up at <https://huggingface.co/join>.
2. Click your avatar → **New Space**.
3. Name: `context-aware-qa`. SDK: **Docker** → template **Blank**. Hardware: **CPU basic (free)**. Visibility: **Public** (a private Space's app is visible only to you when logged in).
4. Click **Create Space**.

### B.2 Add the Gemini key as a secret
In the Space: **Settings** → **Variables and secrets** → **New secret**: name `GEMINI_API_KEY`, value = your key. Optionally add variables such as `GEMINI_MODEL`.

### B.3 Get a Hugging Face access token
Create a token with **Write** permission at <https://huggingface.co/settings/tokens>. Git uses it as the password when you push.

### B.4 Push a deploy-only copy of the code
Hugging Face rejects ordinary git pushes that contain binary files (this repo has PDFs and PNG screenshots), and it reads the Space's settings from a YAML header at the top of `README.md`. So push a separate one-commit branch that drops the binaries and adds the header; your GitHub `main` stays unchanged.

```bash
git remote add space https://huggingface.co/spaces/<your-hf-username>/context-aware-qa   # first time only

git checkout --orphan hf-space            # new branch with no history
git rm -r -q --cached docs/pdf screenshots
{ printf -- '---\ntitle: Context-Aware Q&A Assistant\nemoji: 📘\ncolorFrom: blue\ncolorTo: indigo\nsdk: docker\napp_port: 8000\npinned: false\n---\n\n'; cat README.md; } > README.tmp && mv README.tmp README.md
git add README.md
git commit -m "Deploy to Hugging Face Spaces"
git push --force space hf-space:main

git checkout -f main                      # back to your normal branch (restores README.md)
git branch -D hf-space
```
`app_port: 8000` tells the Space which port the container listens on (the `Dockerfile` starts uvicorn on 8000).

### B.5 Wait for the build and test
The Space page shows **Building** → **Running** (3–5 minutes); click **Logs** to follow it. The app appears on the Space page and directly at `https://<your-hf-username>-context-aware-qa.hf.space`. Check `/api/health` as in A.4.

### B.6 Updating
Commit and push to GitHub as usual, then repeat the commands in B.4 (skip `git remote add`).

---

## Option C — Your own Linux server (Ubuntu VM)

Oracle Cloud's **Always Free** tier gives a small VM that stays on permanently (credit card required for verification, not charged). Any Ubuntu 22.04/24.04 server works the same way.

### C.1 Create the VM
1. Sign up at <https://www.oracle.com/cloud/free/>.
2. **Compute → Instances → Create instance**. Image: **Ubuntu 24.04**. Shape: an *Always Free eligible* shape (e.g. `VM.Standard.A1.Flex` with 1 OCPU / 6 GB, or `VM.Standard.E2.1.Micro`).
3. Download the generated SSH private key, then **Create**.
4. Open ports 80 and 443: on the instance's **Subnet → Security List → Add Ingress Rules**, source `0.0.0.0/0`, TCP, destination ports `80,443`.
5. Note the instance's **public IP**.

### C.2 Connect and install system packages
```bash
chmod 600 ~/Downloads/ssh-key.key
ssh -i ~/Downloads/ssh-key.key ubuntu@<public-ip>

sudo apt update && sudo apt -y upgrade
sudo apt -y install python3 python3-venv git
# Oracle's Ubuntu images block ports in iptables as well; open 80 and 443:
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo apt -y install iptables-persistent && sudo netfilter-persistent save
```

### C.3 Install the app
```bash
sudo mkdir -p /opt/qa && sudo chown ubuntu:ubuntu /opt/qa
git clone https://github.com/ranakaushik/context-aware-qa.git /opt/qa/app
cd /opt/qa/app
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
nano .env          # set GEMINI_API_KEY=...
chmod 600 .env
.venv/bin/python -m pytest -q
```
For a private GitHub repository, clone with a GitHub personal access token or a deploy key.

### C.4 Run it as a service (starts on boot, restarts on crash)
Create `/etc/systemd/system/qa.service`:
```ini
[Unit]
Description=Context-Aware Q&A Assistant
After=network-online.target

[Service]
User=ubuntu
WorkingDirectory=/opt/qa/app
ExecStart=/opt/qa/app/.venv/bin/uvicorn qa_assistant.api:app --host 127.0.0.1 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now qa
systemctl status qa                 # should say "active (running)"
curl -s localhost:8000/api/health
```

### C.5 Put HTTPS in front with Caddy
```bash
sudo apt -y install caddy
```
Edit `/etc/caddy/Caddyfile`:
- **With a domain** (point an `A` record at the public IP first; free subdomains are available from e.g. DuckDNS):
  ```
  qa.example.com {
      reverse_proxy 127.0.0.1:8000
  }
  ```
  Caddy obtains an HTTPS certificate automatically.
- **Without a domain** (plain HTTP on the IP):
  ```
  :80 {
      reverse_proxy 127.0.0.1:8000
  }
  ```
```bash
sudo systemctl reload caddy
```
Open `https://qa.example.com` (or `http://<public-ip>`).

### C.6 Updating
```bash
cd /opt/qa/app && git pull && .venv/bin/pip install -r requirements.txt && sudo systemctl restart qa
```
Logs: `journalctl -u qa -f`.

---

## After deploying: protect your Gemini quota

A public URL means anyone can ask questions, and each question uses your Gemini key. Off-topic questions are rejected without a model call, but on-topic ones are not limited.
- A free-tier Gemini key (no billing enabled) cannot cost money; when its quota runs out, users see the "rate limit or quota reached" message until it resets.
- If you enable billing on the Google project, set a budget alert in Google Cloud Console → **Billing → Budgets & alerts**.
- To demo without a key, set `QA_MODE=offline`.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Build fails on `pip install` | Python version too new or old for a package — keep `PYTHON_VERSION=3.12.7` (Render) / `python:3.12-slim` (Docker). |
| Render: "No open ports detected" / deploy times out | Start command must use `--host 0.0.0.0 --port $PORT`. |
| Spaces: stuck on "Starting" | `app_port: 8000` missing from the `README.md` YAML header pushed in B.4. |
| Spaces: push rejected, "binary files" | A PDF/PNG is still in the pushed branch — repeat B.4, which removes `docs/pdf` and `screenshots`. |
| `/api/health` shows `"mode":"offline"` | `GEMINI_API_KEY` not set on the host (check spelling), or service not redeployed after adding it. |
| "Gemini rejected the credentials" | Wrong or revoked key — create a new one in AI Studio and update the secret. |
| "Gemini is unavailable right now (error 503/504)" or slow answers | Google's API is overloaded. Try again later, or set `GEMINI_MODEL=gemini-flash-lite-latest`. |
| First page load takes ~1 minute (Render) | The free service was asleep — see A.6. |
| 502 from Caddy (option C) | The app isn't running: `systemctl status qa`, `journalctl -u qa -n 50`. |

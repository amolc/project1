# OpenRouter Chat CLI

A minimal Python command-line client for [OpenRouter](https://openrouter.ai) chat completions. It sends a prompt and prints the model's reply.

Repository: <https://github.com/amolc/project1>

## Requirements

- Python 3.9+ (tested on 3.14)
- An OpenRouter account and API key

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/amolc/project1.git
cd project1
```

### 2. Create and activate a virtual environment

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get an OpenRouter API key

1. Sign in at <https://openrouter.ai>.
2. Open the keys page: <https://openrouter.ai/keys>.
3. Click **Create Key**, give it a name, and copy the generated key (`sk-or-...`).
4. Add credits if needed under **Credits**; many models also have free variants.

Keep the key secret. Never commit it to Git.

### 5. Configure

Copy the example environment file and fill in your key:

```bash
cp .env.example .env
```

Edit `.env`:

```env
OPENROUTER_API_KEY=sk-or-your-key-here
OPENROUTER_MODEL=openai/gpt-4o-mini
```

Supported variables:

| Variable             | Required | Default                          | Description                              |
| -------------------- | -------- | -------------------------------- | ---------------------------------------- |
| `OPENROUTER_API_KEY` | yes      | —                                | Your API key from openrouter.ai/keys     |
| `OPENROUTER_MODEL`   | no       | `openai/gpt-4o-mini`           | Any model ID listed on OpenRouter        |
| `OPENROUTER_API_URL` | no       | `https://openrouter.ai/api/v1/chat/completions` | API endpoint override |

You can browse available model IDs at <https://openrouter.ai/models>.

## Usage

Pass the prompt as arguments:

```bash
python openrouter_chat.py "Explain what a race condition is in one paragraph"
```

Or pipe it through stdin:

```bash
echo "Write a haiku about Python" | python openrouter_chat.py
```

The reply is printed to stdout. Errors from the API are printed to stderr.

## Troubleshooting

**`SSL: CERTIFICATE_VERIFY_FAILED`** — the script loads CA certificates from `certifi`, so make sure dependencies are installed in the active virtual environment. If it still fails, reinstall:

```bash
pip install --force-reinstall certifi
```

**`OPENROUTER_API_KEY is not set`** — confirm `.env` exists in the project root (not just `.env.example`) and contains your key.

**HTTP 401** — the key is invalid, revoked, or has no credits.

**HTTP 402** — the account has insufficient credits for the selected model.

## .gitignore

`.env`, `.venv/`, and `__pycache__/` are ignored so secrets and local environments stay out of the repository.

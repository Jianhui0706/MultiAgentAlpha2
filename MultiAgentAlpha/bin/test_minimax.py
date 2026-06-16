import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def load_env():
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                key, value = key.strip(), value.strip()
                if key not in os.environ:
                    os.environ[key] = value

load_env()

api_key = os.getenv("OPENAI_API_KEY", "")
base_url = os.getenv("OPENAI_BASE_URL", "")
model = os.getenv("LLM_MODEL", "")

print(f"API Key: {api_key[:10]}..." if api_key else "API Key: NOT SET")
print(f"Base URL: {base_url}")
print(f"Model: {model}")
print()

from openai import OpenAI

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

print("Testing MiniMax API...")
try:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello, MiniMax!' and nothing else."},
        ],
        max_tokens=50,
    )
    print(f"Success! Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"Error: {e}")

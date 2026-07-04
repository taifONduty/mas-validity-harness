"""Chat backbone — one client for any OpenAI-compatible provider.

Providers:
  zai          General z.ai API key  -> https://api.z.ai/api/paas/v4
  zai-coding   GLM Coding Plan key   -> https://api.z.ai/api/coding/paas/v4
  openrouter   OpenRouter key        -> https://openrouter.ai/api/v1

Env vars: ZAI_API_KEY or OPENROUTER_API_KEY (or pass api_key=...).

Free GLM models on z.ai: glm-4.5-flash, glm-4.7-flash (rate-limited ->
use sleep_s=1-2 in run_many). Your whole study can run at $0.

SCIENCE NOTE (do not skip): GLM models support a 'thinking' mode that emits
reasoning before the answer. We DISABLE it explicitly, for two reasons:
(1) the metrics measure the agent's visible turn text — mixing hidden
reasoning in/out across turns contaminates every distribution and embedding;
(2) the anchor paper found reasoning modes may WORSEN sharded performance,
which is a separate manipulated variable, not something to leave floating.
Keep thinking disabled for all main runs; a thinking-on arm can be a
controlled follow-up experiment later.
"""
import json
import os
import time
import urllib.request

PROVIDERS = {
    "zai": {
        "url": "https://api.z.ai/api/paas/v4/chat/completions",
        "key_env": "ZAI_API_KEY",
    },
    "zai-coding": {
        "url": "https://api.z.ai/api/coding/paas/v4/chat/completions",
        "key_env": "ZAI_API_KEY",
    },
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "key_env": "OPENROUTER_API_KEY",
    },
}

# sensible defaults per provider
DEFAULT_MODELS = {
    "zai": "glm-4.5-flash",            # free; swap to glm-4.7-flash / glm-5.x later
    "zai-coding": "glm-4.7",
    "openrouter": "meta-llama/llama-3.1-8b-instruct",
}


class ChatBackbone:
    def __init__(self, provider="zai", model=None, temperature=1.0,
                 max_tokens=1000, api_key=None, disable_thinking=True):
        cfg = PROVIDERS[provider]
        self.provider = provider
        self.url = cfg["url"]
        self.model = model or DEFAULT_MODELS[provider]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.disable_thinking = disable_thinking
        self.api_key = api_key or os.environ.get(cfg["key_env"])
        if not self.api_key:
            raise RuntimeError(f"Set {cfg['key_env']} in your environment.")

    def chat(self, messages, retries=4):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.provider.startswith("zai") and self.disable_thinking:
            payload["thinking"] = {"type": "disabled"}
        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
        )
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    data = json.loads(r.read().decode())
                msg = data["choices"][0]["message"]
                content = msg.get("content") or ""
                if not content.strip():
                    raise ValueError("empty content (thinking mode leak?)")
                return content
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)


# Back-compat aliases used elsewhere in the repo
def OpenRouterBackbone(model=None, **kw):
    return ChatBackbone(provider="openrouter", model=model, **kw)

REPLICATION_MODEL = "deepseek/deepseek-chat"


class MockBackbone:
    """Deterministic fake for pipeline dry-runs (no API, no cost)."""
    def __init__(self, model="mock"):
        self.model = model
        self._i = 0

    def chat(self, messages):
        self._i += 1
        role_hint = messages[0]["content"][:40].lower()
        if "plan" in role_hint:
            return f"Plan v{self._i}: restate constraints, decompose, assign steps."
        if "critic" in role_hint:
            return f"Critique v{self._i}: check constraints; found no issue #{self._i}."
        return f"Attempt v{self._i}: FINAL ANSWER: 42"

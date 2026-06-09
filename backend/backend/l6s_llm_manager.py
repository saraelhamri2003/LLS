import json
import os
import re
import urllib.error
import urllib.request


def _normalize_api_model_name(model_name):
    """Normalize human-readable model labels to provider-friendly IDs."""
    if model_name is None:
        return ""

    raw = str(model_name).strip()
    if not raw:
        return ""

    lowered = raw.lower()
    aliases = {
        "gemini 3 pro": "gemini-3-pro",
        "gemini 3 flash": "gemini-3-flash",
        "gemini 2.5 pro": "gemini-2.5-pro",
        "gemini 2.5 flash": "gemini-2.5-flash",
        "gemini 2.5 flash-lite": "gemini-2.5-flash-lite",
        "gpt-5.2 instant": "gpt-5.2-instant",
        "gpt-5.2 thinking": "gpt-5.2-thinking",
        "gpt-5.2 pro": "gpt-5.2-pro",
        "gpt-5.1 instant": "gpt-5.1-instant",
        "gpt-5.1 thinking": "gpt-5.1-thinking",
        "gpt-4.1 mini": "gpt-4.1-mini",
        "gpt-4.1 nano": "gpt-4.1-nano",
        "claude opus 4.5": "claude-opus-4-5",
        "claude sonnet 4.5": "claude-sonnet-4-5",
        "claude haiku 4.5": "claude-haiku-4-5",
    }
    if lowered in aliases:
        return aliases[lowered]

    # Keep already machine-friendly IDs untouched.
    if " " not in raw and raw == lowered:
        return raw

    normalized = lowered.replace("_", "-")
    normalized = re.sub(r"\s+", "-", normalized)
    normalized = re.sub(r"[^a-z0-9._:/-]", "-", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
    return normalized or raw


class _OllamaHttpClient:
    def __init__(self, model, base_url, temperature):
        self.model = model
        self.base_url = (base_url or "http://127.0.0.1:11434").rstrip("/")
        self.temperature = temperature

    def invoke(self, prompt):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if self.temperature is not None:
            payload["options"] = {"temperature": self.temperature}

        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode("utf-8")
        result = json.loads(body)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result.get("response", "")


class LLMManager:
    def __init__(
        self,
        local_model="llama3:8b",
        api_model="gemini-pro",
        temperature=0.3,
        api_key=None,
    ):
        self.local_model = local_model
        self.requested_api_model = api_model
        self.api_model = _normalize_api_model_name(api_model)
        self.temperature = temperature
        self.api_key = api_key

        self._local_llm = self._init_local()
        self._api_llm = self._init_api()

        if self._local_llm:
            self.active = "local"
        elif self._api_llm:
            self.active = "api"
        else:
            self.active = "none"

    def is_available(self):
        return self._local_llm is not None or self._api_llm is not None

    def switch_to_local(self):
        if self._local_llm is None:
            return False
        self.active = "local"
        return True

    def switch_to_api(self):
        if self._api_llm is None:
            return False
        self.active = "api"
        return True

    def get_available_models(self):
        return {
            "local": {"available": self._local_llm is not None, "model": self.local_model},
            "api": {
                "available": self._api_llm is not None,
                "model": self.api_model,
                "requested_model": self.requested_api_model,
            },
            "active": self.active,
        }

    def get_active_model_info(self):
        if self.active == "local":
            return f"local:{self.local_model}"
        if self.active == "api":
            return f"api:{self.api_model}"
        return "none"

    def _invoke_llm(self, llm, prompt):
        result = llm.invoke(prompt) if hasattr(llm, "invoke") else llm(prompt)
        if hasattr(result, "content"):
            return result.content
        return str(result)

    def invoke(self, prompt):
        if self.active == "local":
            candidates = [("local", self._local_llm), ("api", self._api_llm)]
        elif self.active == "api":
            candidates = [("api", self._api_llm), ("local", self._local_llm)]
        else:
            candidates = [("local", self._local_llm), ("api", self._api_llm)]

        for name, llm in candidates:
            if llm is None:
                continue
            try:
                result_text = self._invoke_llm(llm, prompt)
                if name != self.active:
                    self.active = name
                return result_text
            except Exception:
                continue

        return self._fallback_response()

    def invoke_api_only(self, prompt):
        if self._api_llm is None:
            raise RuntimeError("API model is not available.")
        result_text = self._invoke_llm(self._api_llm, prompt)
        self.active = "api"
        return result_text

    def invoke_local_only(self, prompt):
        if self._local_llm is None:
            raise RuntimeError("Local model is not available.")
        result_text = self._invoke_llm(self._local_llm, prompt)
        self.active = "local"
        return result_text

    def _init_local(self):
        base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        kwargs = {"model": self.local_model, "temperature": self.temperature}
        if base_url:
            kwargs["base_url"] = base_url

        try:
            from langchain_community.llms import Ollama
            return Ollama(**kwargs)
        except Exception:
            return _OllamaHttpClient(self.local_model, base_url, self.temperature)

    def _init_api(self):
        api_model_lower = (self.api_model or "").lower()
        api_key = self.api_key

        if "gemini" in api_model_lower:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except Exception:
                return None
            key = api_key or os.getenv("GOOGLE_API_KEY")
            try:
                return ChatGoogleGenerativeAI(
                    model=self.api_model,
                    temperature=self.temperature,
                    google_api_key=key,
                )
            except Exception:
                return None

        if "claude" in api_model_lower:
            try:
                from langchain_anthropic import ChatAnthropic
            except Exception:
                return None
            key = api_key or os.getenv("ANTHROPIC_API_KEY")
            try:
                return ChatAnthropic(
                    model=self.api_model,
                    temperature=self.temperature,
                    api_key=key,
                )
            except Exception:
                return None

        if any(token in api_model_lower for token in ("gpt", "openai", "codex", "vision", "audio")) or re.fullmatch(
            r"o[1-9](?:[._-].*)?",
            api_model_lower,
        ):
            try:
                from langchain_openai import ChatOpenAI
            except Exception:
                return None
            key = api_key or os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")
            try:
                kwargs = {
                    "model": self.api_model,
                    "temperature": self.temperature,
                    "api_key": key,
                }
                if base_url:
                    kwargs["base_url"] = base_url
                return ChatOpenAI(**kwargs)
            except Exception:
                return None

        return None

    def _fallback_response(self):
        return (
            "LLM is not configured. Provide CSF values for predictions, or configure "
            "OLLAMA_BASE_URL or an API key for a supported model."
        )

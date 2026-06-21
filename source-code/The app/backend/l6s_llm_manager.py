import json
import os
import urllib.error
import urllib.request


def is_ollama_running(base_url):
    try:
        import urllib.request
        url = (base_url or "http://127.0.0.1:11434").rstrip("/")
        with urllib.request.urlopen(f"{url}/api/tags", timeout=1.0) as response:
            return response.status == 200
    except Exception:
        return False

def map_model_name(model_name):
    name = (model_name or "").lower().strip()
    if "gemini" in name:
        if "flash" in name:
            return "gemini-1.5-flash"
        return "gemini-1.5-pro"
    if "gpt" in name or "openai" in name:
        if "mini" in name or "instant" in name or "nano" in name:
            return "gpt-4o-mini"
        return "gpt-4o"
    if "claude" in name:
        if "haiku" in name:
            return "claude-3-haiku-20240307"
        return "claude-3-5-sonnet-latest"
    return model_name

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
        self.api_model = api_model
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
            "api": {"available": self._api_llm is not None, "model": self.api_model},
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
        if not is_ollama_running(base_url):
            return None
        
        kwargs = {"model": self.local_model, "temperature": self.temperature}
        if base_url:
            kwargs["base_url"] = base_url

        try:
            from langchain_community.llms import Ollama
            return Ollama(**kwargs)
        except Exception:
            return _OllamaHttpClient(self.local_model, base_url, self.temperature)

    def _init_api(self):
        mapped_model = map_model_name(self.api_model)
        api_model_lower = (mapped_model or "").lower()
        api_key = self.api_key

        if "gemini" in api_model_lower:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except Exception:
                return None
            key = api_key or os.getenv("GOOGLE_API_KEY")
            try:
                return ChatGoogleGenerativeAI(
                    model=mapped_model,
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
                    model=mapped_model,
                    temperature=self.temperature,
                    api_key=key,
                )
            except Exception:
                return None

        if any(token in api_model_lower for token in ("gpt", "openai", "codex", "vision", "audio")):
            try:
                from langchain_openai import ChatOpenAI
            except Exception:
                return None
            key = api_key or os.getenv("OPENAI_API_KEY")
            try:
                return ChatOpenAI(
                    model=mapped_model,
                    temperature=self.temperature,
                    api_key=key,
                )
            except Exception:
                return None

        return None

    def _fallback_response(self):
        return (
            "🤖 **LLM Connection Offline**\n\n"
            "To enable the AI Chatbot to respond to your queries, please do one of the following:\n\n"
            "1. **Configure in UI**: Click the **Settings Gear Icon** ⚙️ at the top of the chat screen, toggle **Use API Model**, enter your **Gemini / OpenAI API Key**, and select a model (e.g., Gemini 3 Flash).\n"
            "2. **Configure via Backend Environment**: Create a `.env` file in `source-code/The app/backend/.env` with your API key:\n"
            "   ```env\n"
            "   GOOGLE_API_KEY=your_gemini_key\n"
            "   ```\n"
            "   *(or set `OPENAI_API_KEY` for OpenAI, or `ANTHROPIC_API_KEY` for Anthropic)*\n"
            "3. **Run Ollama Locally**: If you prefer a local model, ensure Ollama is running on your machine (default port 11434) with `llama3:8b` downloaded (`ollama pull llama3:8b`).\n\n"
            "👉 *Note: You can still run the performance engine and recommendations without an LLM! Just fill out the **Guided Assessment** or parameters on the sidebar and click **Calculate Performance**.*"
        )

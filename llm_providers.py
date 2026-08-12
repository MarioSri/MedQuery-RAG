"""Provider-independent LLM interface (roadmap items 2/3).

LLMProvider is the abstract contract; the RAG engine calls
``llm.generate(prompt)`` and never imports a vendor SDK directly.

Supported providers:
  - anthropic:   Claude via the anthropic SDK
  - openai:      GPT via the openai SDK
  - gemini:      Google Gemini via google-genai
  - ollama:      local Ollama models
  - custom:      any OpenAI-compatible endpoint (vLLM, LM Studio, ...)
"""
import config


class LLMProvider:
    """Abstract LLM interface."""

    def generate(self, prompt: str, system: str = "", **kwargs) -> dict:
        """Return {"text": str, "usage": {"input_tokens": int, "output_tokens": int}}."""
        raise NotImplementedError

    @property
    def model_id(self) -> str:
        raise NotImplementedError


class AnthropicProvider(LLMProvider):
    def __init__(self):
        import anthropic
        self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    def generate(self, prompt, system="", **kwargs):
        resp = self._client.messages.create(
            model=config.LLM_MODEL,
            max_tokens=config.LLM_MAX_TOKENS,
            temperature=config.LLM_TEMPERATURE,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return {
            "text": resp.content[0].text,
            "usage": {"input_tokens": resp.usage.input_tokens,
                      "output_tokens": resp.usage.output_tokens},
        }

    @property
    def model_id(self):
        return f"anthropic/{config.LLM_MODEL}"


class OpenAIProvider(LLMProvider):
    def __init__(self):
        from openai import OpenAI
        self._client = OpenAI(api_key=config.OPENAI_API_KEY)

    def generate(self, prompt, system="", **kwargs):
        resp = self._client.chat.completions.create(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
            messages=(
                [{"role": "system", "content": system}] if system else []
            ) + [{"role": "user", "content": prompt}],
        )
        u = resp.usage
        return {
            "text": resp.choices[0].message.content,
            "usage": {"input_tokens": u.prompt_tokens, "output_tokens": u.completion_tokens},
        }

    @property
    def model_id(self):
        return f"openai/{config.LLM_MODEL}"


class GeminiProvider(LLMProvider):
    def __init__(self):
        from google import genai
        self._client = genai.Client(api_key=config.GEMINI_API_KEY)

    def generate(self, prompt, system="", **kwargs):
        resp = self._client.models.generate_content(
            model=config.LLM_MODEL,
            contents=prompt,
            config={"system_instruction": system} if system else {},
        )
        return {"text": resp.text, "usage": {"input_tokens": 0, "output_tokens": 0}}

    @property
    def model_id(self):
        return f"gemini/{config.LLM_MODEL}"


class OllamaProvider(LLMProvider):
    def __init__(self):
        from ollama import Client
        self._client = Client(host=config.OLLAMA_BASE_URL)

    def generate(self, prompt, system="", **kwargs):
        resp = self._client.chat(
            model=config.LLM_MODEL,
            messages=([{"role": "system", "content": system}] if system else [])
                     + [{"role": "user", "content": prompt}],
        )
        u = resp.get("prompt_eval_count", 0), resp.get("eval_count", 0)
        return {
            "text": resp["message"]["content"],
            "usage": {"input_tokens": u[0], "output_tokens": u[1]},
        }

    @property
    def model_id(self):
        return f"ollama/{config.LLM_MODEL}"


class CustomOpenAICompatibleProvider(LLMProvider):
    """Any OpenAI-compatible endpoint (vLLM, LM Studio, local servers)."""

    def __init__(self):
        from openai import OpenAI
        self._client = OpenAI(base_url=config.CUSTOM_BASE_URL, api_key=config.CUSTOM_API_KEY)

    def generate(self, prompt, system="", **kwargs):
        resp = self._client.chat.completions.create(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
            messages=(
                [{"role": "system", "content": system}] if system else []
            ) + [{"role": "user", "content": prompt}],
        )
        u = resp.usage
        return {
            "text": resp.choices[0].message.content,
            "usage": {"input_tokens": u.prompt_tokens, "output_tokens": u.completion_tokens},
        }

    @property
    def model_id(self):
        return f"custom/{config.LLM_MODEL}"


PROVIDERS = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "ollama": OllamaProvider,
    "custom": CustomOpenAICompatibleProvider,
}


def get_llm_provider() -> LLMProvider:
    name = config.LLM_PROVIDER.lower()
    if name not in PROVIDERS:
        raise ValueError(f"Unknown LLM provider: {name}. Choose from {sorted(PROVIDERS)}")
    return PROVIDERS[name]()

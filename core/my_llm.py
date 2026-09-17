import os
from typing import Optional
from openai import OpenAI
from hello_agents import HelloAgentsLLM, HelloAgentsException


class MyLLM(HelloAgentsLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = "auto",
        **kwargs,
    ):
        if provider in (None, "auto"):
            provider = self._auto_detect_provider(api_key, base_url)

        if provider == "wc":
            print("正在使用自定义的 WC Provider")
            self.provider = "wc"

            self.api_key = api_key or os.getenv("WC_API_KEY")
            self.base_url = base_url or os.getenv("WC_BASE_URL")

            # 验证凭证是否存在
            if not self.api_key:
                raise ValueError(
                    "WC API key not found. Please set WC_API_KEY environment variable."
                )

            # 设置默认模型和其他参数
            self.model = (
                model
                or os.getenv("WC_MODEL_NAME")
                or os.getenv("LLM_MODEL_ID")
                or "qwen3.5-flash"
            )
            self.temperature = kwargs.get("temperature", 0.7)
            self.max_tokens = kwargs.get("max_tokens")
            self.timeout = kwargs.get("timeout", 60)

            # 使用获取的参数创建OpenAI客户端实例
            self._client = OpenAI(
                api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
            )

        elif provider == "gemini":
            self.provider = "gemini"
            self.api_key = api_key or os.getenv("GEMINI_API_KEY")
            self.base_url = (
                base_url
                or os.getenv("GEMINI_BASE_URL")
                or "https://generativelanguage.googleapis.com/v1beta/openai/"
            )
            if not self.api_key:
                raise ValueError(
                    "Gemini API key not found. Please set GEMINI_API_KEY environment variable."
                )
            self.model = model or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"
            self.temperature = kwargs.get("temperature", 0.7)
            self.max_tokens = kwargs.get("max_tokens")
            self.timeout = kwargs.get("timeout", 60)
            self._client = OpenAI(
                api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
            )
        else:
            super().__init__(
                model=model,
                api_key=api_key,
                base_url=base_url,
                provider=provider,
                **kwargs,
            )

    def _auto_detect_provider(self, api_key, base_url):
        if os.getenv("WC_API_KEY"):
            return "wc"
        if os.getenv("GEMINI_API_KEY"):
            return "gemini"
        if "generativelanguage.googleapis.com" in (base_url or "").lower():
            return "gemini"
        return super()._auto_detect_provider(api_key, base_url)

    def think(self, messages, temperature=None):
        """流式调用：跳过服务商在流末尾发送的空 choices 块（如 usage-only chunk）。"""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )
            for chunk in response:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content or ""
                if content:
                    print(content, end="", flush=True)
                    yield content
            print()
        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            raise HelloAgentsException(f"LLM调用失败: {str(e)}")

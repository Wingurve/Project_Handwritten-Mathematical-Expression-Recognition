"""DeepSeek client for LaTeX post-processing."""
import os

from dotenv import load_dotenv

load_dotenv()


class DeepSeekEmbedder:
    """OpenAI-compatible DeepSeek wrapper used to polish BTTR LaTeX output.

    The class name is kept for compatibility with the existing inference code.
    """

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
    ):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = (base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.client = None
        if self.api_key:
            self._init_client()

    def _init_client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Please install openai: pip install openai") from exc

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def correct_with_deepseek(
        self,
        text: str,
        instruction: str = "请修正以下 LaTeX 数学公式中的明显识别错误，只输出可直接渲染的 LaTeX 代码，不要解释，不要使用 Markdown，不要添加美元符号：",
    ) -> str:
        if self.client is None:
            raise ValueError("DeepSeek API key is not configured")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise LaTeX post-processor for handwritten mathematical expression recognition.",
                },
                {"role": "user", "content": f"{instruction}\n\n{text}"},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()

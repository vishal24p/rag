"""Direct answers using the existing generator model and its cached client."""
from src.config import settings

DIRECT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's request directly and concisely. "
    "Be honest about uncertainty. Do not invent sources or citations."
)


def build_llm():
    from src.llm.factory import build_llm as existing_client
    return existing_client()


class DirectGenerator:
    def __init__(self):
        self.llm = build_llm()

    def generate(self, query: str) -> str:
        from llama_index.core.llms import ChatMessage, MessageRole
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=DIRECT_SYSTEM_PROMPT),
            ChatMessage(role=MessageRole.USER, content=query),
        ]
        try:
            return str(self.llm.chat(messages))
        except Exception:
            return str(self.llm.complete(
                f"{DIRECT_SYSTEM_PROMPT}\n\n{query}",
                temperature=settings.gen_temperature,
                max_tokens=settings.gen_max_tokens,
            ))

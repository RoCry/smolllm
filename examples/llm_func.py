import asyncio
from functools import partial

from smolllm import LLMFunction, ask_llm

# Create a custom LLM function with specific configuration
custom_llm_with_args = partial(
    ask_llm,
    api_key="pollinations_dont_need_api_key",
    model="openai/openai-large",
    base_url="https://text.pollinations.ai/openai#",
)


def translate(llm: LLMFunction, text: str, to: str = "Chinese"):
    return llm(f"Explain the following text in {to}:\n{text}")


async def main():
    print(await translate(custom_llm_with_args, "Show me the money"))


if __name__ == "__main__":
    asyncio.run(main())

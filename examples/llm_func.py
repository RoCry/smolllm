import asyncio
from functools import partial

from smolllm import LLMFunction, ask_llm

# Create a custom LLM function with specific configuration
custom_llm_with_args = partial(
    ask_llm,
    api_key="pollinations_dont_need_api_key",
    # GET https://text.pollinations.ai/models
    model="openai/openai-fast",
    base_url="https://text.pollinations.ai/openai#",
)


def translate(llm: LLMFunction, text: str, to: str = "Chinese"):
    return llm(f"Explain the following text in {to}:\n{text}")


async def main():
    response = await translate(custom_llm_with_args, "Show me the money")
    print(f"response: {response}")
    print(f"model: {response.model}")
    print(f"model_name: {response.model_name}")
    print(f"provider: {response.provider}")


if __name__ == "__main__":
    asyncio.run(main())

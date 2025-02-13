import asyncio
from smolllm import LLMFunction, ask_llm

from functools import partial

custom_ollama = partial(
    ask_llm,
    model="ollama/deepseek-r1:7b",
)


def translate(llm: LLMFunction, text: str, to: str = "Chinese"):
    return llm(f"Translate below text to {to}:\n{text}")


async def main():
    print(await translate(custom_ollama, "Say hello world in a creative way"))


if __name__ == "__main__":
    asyncio.run(main())

import asyncio
from dotenv import load_dotenv
from smolllm import LLMFunction, ask_llm
from functools import partial

# Load environment variables at application startup
load_dotenv()

# Create a custom LLM function with specific configuration
custom_ollama = partial(
    ask_llm,
    # model="ollama/deepseek-r1:7b",
    # base_url="http://rocry-win.local:11434",
)


def translate(llm: LLMFunction, text: str, to: str = "Chinese"):
    return llm(f"Explain the following text in {to}:\n{text}")


async def main():
    print(await translate(custom_ollama, "Show me the money"))


if __name__ == "__main__":
    asyncio.run(main())

import asyncio
import sys

from dotenv import load_dotenv

from smolllm import LLMResponse, ask_llm

_ = load_dotenv()


async def main() -> None:
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Hello"
    response: LLMResponse = await ask_llm(prompt)
    if response.reasoning:
        print(f"[Reasoning]\n{response.reasoning}\n")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())

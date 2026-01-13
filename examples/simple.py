import asyncio

from dotenv import load_dotenv

from smolllm import LLMResponse, ask_llm

_ = load_dotenv()


async def main(prompt: str = "Hello") -> None:
    response: LLMResponse = await ask_llm(prompt)
    print(response)


if __name__ == "__main__":
    asyncio.run(main())

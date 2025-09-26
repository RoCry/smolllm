import asyncio

from dotenv import load_dotenv

from smolllm import LLMResponse, ask_llm

_ = load_dotenv()


async def main(prompt: str = "Say hello world in a creative way") -> None:
    response: LLMResponse = await ask_llm(prompt)
    print(response)


if __name__ == "__main__":
    asyncio.run(main())

import asyncio
from smolllm import ask_llm


async def simple():
    print(
        await ask_llm(
            "Say hello world in a creative way", model="gemini/gemini-2.0-flash"
        )
    )


async def main():
    await simple()


if __name__ == "__main__":
    asyncio.run(main())

import asyncio

from dotenv import load_dotenv

from smolllm import stream_llm

load_dotenv()


async def main(prompt: str = "Say hello world in a creative way"):
    response = stream_llm(
        prompt,
        # model="gemini/gemini-2.0-flash",  # specify model can override env.SMOLLLM_MODEL
    )
    async for r in response:
        print(r, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

import asyncio

from dotenv import load_dotenv

from smolllm import stream_llm

load_dotenv()


async def main(prompt: str = "Say hello world in a creative way"):
    response = await stream_llm(
        prompt,
        # model="gemini/gemini-2.0-flash",  # specify model can override env.SMOLLLM_MODEL
        # model=[
        #     "gemini/gemini-2.5-pro-exp-03-25",  # free account can not use this model, so it will fallback to the next model
        #     "gemini/gemini-2.5-flash-preview-05-20",
        # ],
    )
    # response is now a StreamResponse object with model info
    print(f"Streaming from model: {response.model}")
    async for r in response:
        print(r, end="", flush=True)
    print()  # New line after streaming


if __name__ == "__main__":
    asyncio.run(main())

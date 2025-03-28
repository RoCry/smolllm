import argparse
import asyncio

from dotenv import load_dotenv

from smolllm import stream_llm


async def simple(prompt: str = "Say hello world in a creative way"):
    response = stream_llm(
        prompt,
        # model="gemini/gemini-2.0-flash",  # specify model can override env.SMOLLLM_MODEL
    )
    async for r in response:
        print(r, end="", flush=True)


async def main():
    # Load environment variables at application startup
    load_dotenv()

    parser = argparse.ArgumentParser(description="Ask LLM for creative responses")
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Say hello world in a creative way",
        help="Custom prompt to send to LLM",
    )

    args = parser.parse_args()
    await simple(args.prompt)


if __name__ == "__main__":
    asyncio.run(main())

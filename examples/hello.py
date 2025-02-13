import asyncio
import argparse
from dotenv import load_dotenv
from smolllm import ask_llm


async def simple(prompt: str = "Say hello world in a creative way"):
    response = await ask_llm(
        prompt,
        model="gemini/gemini-2.0-flash",  # Default model can be overridden by SMOLLLM_MODEL
    )
    print(response)


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

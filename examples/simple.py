import argparse
import asyncio

from dotenv import load_dotenv

from smolllm import LLMResponse, ask_llm

_ = load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", default="Hello")
    parser.add_argument("--reasoning-effort", dest="reasoning_effort")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    response: LLMResponse = await ask_llm(
        args.prompt,
        reasoning_effort=args.reasoning_effort,
    )
    print(response)


if __name__ == "__main__":
    asyncio.run(main())

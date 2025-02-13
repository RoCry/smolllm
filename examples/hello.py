import asyncio
import argparse
from smolllm import ask_llm


async def simple(prompt: str = "Say hello world in a creative way"):
    print(await ask_llm(prompt))


async def main():
    parser = argparse.ArgumentParser(description='Ask LLM for creative responses')
    parser.add_argument('prompt', nargs='?', default="Say hello world in a creative way",
                       help='Custom prompt to send to LLM')
    
    args = parser.parse_args()
    await simple(args.prompt)


if __name__ == "__main__":
    asyncio.run(main())

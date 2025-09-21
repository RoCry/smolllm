import asyncio

from dotenv import load_dotenv

from smolllm import Message, stream_llm

_ = load_dotenv()

prompt: list[Message] = [
    {"role": "user", "content": "Hi, I'm John. Please response as short as possible."},
    {"role": "assistant", "content": "OK"},
    {"role": "user", "content": "How to say my name in Chinese?"},
]


async def main():
    response = await stream_llm(prompt)
    async for r in response:
        print(r, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())

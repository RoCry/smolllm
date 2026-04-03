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
    reasoning_started = False
    content_started = False
    async for chunk in response:
        if chunk.reasoning:
            if not reasoning_started:
                print("[Reasoning]", flush=True)
                reasoning_started = True
            print(chunk.reasoning, end="", flush=True)
        if chunk.content:
            if reasoning_started and not content_started:
                print("\n\n[Answer]", flush=True)
            content_started = True
            print(chunk.content, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())

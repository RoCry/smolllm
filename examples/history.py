import asyncio

from smolllm import Message, stream_llm

prompt: list[Message] = [
    {"role": "user", "content": "Hi, I'm John. Please response as short as possible."},
    {"role": "assistant", "content": "OK"},
    {"role": "user", "content": "How to say my name in Chinese?"},
]


async def main():
    response = await stream_llm(prompt)
    text = await response.display()
    print(text)


if __name__ == "__main__":
    asyncio.run(main())

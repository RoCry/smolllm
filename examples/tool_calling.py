"""Tool calling: smolllm carries the calls, the caller runs the loop.

The library never executes a tool and never inspects the argument JSON — it hands
you the provider's calls verbatim, and you append the results back as messages.
"""

import asyncio
import json

from smolllm import Message, ask_llm

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def get_weather(city: str) -> dict[str, object]:
    return {"city": city, "temp_c": 21, "sky": "clear"}


async def main():
    messages: list[Message] = [{"role": "user", "content": "What's the weather in San Francisco?"}]

    while True:
        response = await ask_llm(messages, extra_body={"tools": TOOLS})
        if not response.tool_calls:
            print(response.text)
            return

        # Replay the assistant turn verbatim, then answer each call.
        messages.append({"role": "assistant", "content": response.text or None, "tool_calls": response.tool_calls})
        for call in response.tool_calls:
            function = call["function"]
            arguments = json.loads(function["arguments"])
            result = get_weather(**arguments)
            messages.append(
                {"role": "tool", "tool_call_id": str(call["id"]), "content": json.dumps(result, ensure_ascii=False)}
            )


if __name__ == "__main__":
    asyncio.run(main())

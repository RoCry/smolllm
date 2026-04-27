import argparse
import asyncio

from dotenv import load_dotenv

from smolllm import embed_llm

_ = load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="*", default=["hello world"])
    parser.add_argument("--model", default="ollama/qwen3-embedding:4b")
    parser.add_argument("--dimensions", type=int, default=None)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    inputs: str | list[str] = args.text[0] if len(args.text) == 1 else list(args.text)
    response = await embed_llm(inputs, model=args.model, dimensions=args.dimensions)
    print(f"model={response.model_name} count={len(response)} dim={response.dimensions}")
    for i, vec in enumerate(response.embeddings):
        preview = ", ".join(f"{x:+.4f}" for x in vec[:5])
        print(f"[{i}] {preview}, ... ({len(vec)} floats)")


if __name__ == "__main__":
    asyncio.run(main())

from smolagents import CodeAgent
from smolllm import SmolllmModel
from dotenv import load_dotenv

load_dotenv()

model = SmolllmModel(model_id="gemini/gemini-2.0-flash")
agent = CodeAgent(tools=[], model=model, add_base_tools=True, max_steps=2)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)
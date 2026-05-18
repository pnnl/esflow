from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider


model = AnthropicModel(
    'claude-haiku-4-5-20251001-v1-project',
    provider=AnthropicProvider(
            api_key='sk-wnvs910MJM5pm89hjZZmmw',
            base_url='https://ai-incubator-api.pnnl.gov'
        )
)
agent = Agent(model, instructions='You are a helpful assistant.')

@agent.tool_plain
def get_weather(city: str) -> str:
    return f'The weather in {city} is sunny'

app = agent.to_web()
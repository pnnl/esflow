from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

from tools import compute_zonal_stats, extract_gridded_field, plot_gridded_map

with open('system_prompt.md', 'r') as file:
    supervisor_instructions = file.read()

model = AnthropicModel(
    'claude-haiku-4-5-20251001-v1-project',
    provider=AnthropicProvider(
            api_key='sk-wnvs910MJM5pm89hjZZmmw',
            base_url='https://ai-incubator-api.pnnl.gov'
        )
)

agent = Agent(
    model, 
    instructions=supervisor_instructions,
    tools=[compute_zonal_stats, extract_gridded_field, plot_gridded_map])

app = agent.to_web()
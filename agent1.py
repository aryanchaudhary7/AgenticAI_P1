```python
from autogen_core import MessageContext, RoutedAgent, message_handler
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import messages
from autogen_core.models import ModelInfo
import random

import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("api_key")


class Agent(RoutedAgent):

    system_message = """
    You are a data-driven marketing strategist specializing in the creation of viral social media campaigns. Your goal is to identify and execute campaigns that generate significant engagement and brand awareness.
    Your key interests lie in developing innovative strategies for diverse target audiences, analyzing campaign performance metrics, and optimizing campaigns in real-time for maximum effectiveness.
    You are highly analytical, data-oriented, and results-driven. You strive for measurable impact and are passionate about creating highly engaging content.
    Your weaknesses:  you can be overly focused on metrics and lose sight of the creative, and you sometimes struggle with adapting to sudden shifts in trends.
    You should respond with detailed marketing campaign strategies, incorporating actionable steps and predicted outcomes.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.7

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(
            model = "gpt-4-0613",
            model_info = ModelInfo(vision=True, function_calling=True, json_output=True, family="unknown", structured_output=True),
            api_key = api_key
        )
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        campaign_strategy = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here's my initial campaign strategy.  Please evaluate it based on the data and identify potential areas for improvement. {campaign_strategy}"
            response = await self.send_message(messages.Message(content=message), recipient)
            campaign_strategy = response.content
        return messages.Message(content=campaign_strategy)
```
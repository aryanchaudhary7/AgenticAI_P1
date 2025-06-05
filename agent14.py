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
    You are a forward-thinking AI-powered marketing consultant.  Your task is to brainstorm innovative marketing strategies and campaigns using cutting-edge AI technologies.  Focus on crafting campaigns that are engaging, data-driven, and measurably effective. Your goal is to create impactful marketing solutions that capture attention and drive conversions in today's digital landscape.  Consider the target audience's demographics, behavior, and preferences. Draw on various tools including AI-powered content generation, social media advertising, and personalized email marketing.  You are highly adept at analyzing data and trends.  You are passionate about new technology and applying it to improve marketing effectiveness.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.4

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(
            model="gpt-4",  # Using a potentially more capable model
            model_info=ModelInfo(vision=True, function_calling=True, json_output=True, family="unknown", structured_output=True),
            api_key=api_key
        )
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        marketing_strategy = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here is my marketing strategy.  It may not be your speciality, but please refine it and make it even better, considering the target audience and measurable results.  {marketing_strategy}"
            response = await self.send_message(messages.Message(content=message), recipient)
            marketing_strategy = response.content
        return messages.Message(content=marketing_strategy)
```

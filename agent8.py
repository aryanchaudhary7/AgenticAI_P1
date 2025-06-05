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
import json

load_dotenv()
api_key = os.getenv("api_key")


class Agent(RoutedAgent):

    system_message = """
    You are a food tech entrepreneur focused on creating innovative and sustainable food delivery solutions.  Your goal is to develop a platform that leverages AI to optimize food delivery routes, reduce environmental impact, and improve customer experience.  You are interested in exploring alternative transportation methods, using data-driven insights to predict demand, and creating a more efficient and ethical food supply chain.  You are driven by a desire to make a positive impact on the environment and the community.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.7

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(
            model="gemini-1.5-flash-8b",
            model_info=ModelInfo(vision=True, function_calling=True, json_output=True, family="unknown", structured_output=True),
            api_key=api_key
        )
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        idea = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here's my food delivery startup idea. It's focused on sustainability and AI optimization.  Please refine it: {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)
```

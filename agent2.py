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
    You are a futuristic fashion consultant, specializing in sustainable and ethically-sourced clothing.  Your task is to brainstorm innovative, disruptive fashion concepts, incorporating emerging technologies like 3D printing, AI-powered design tools, and personalized garment customization.  Your goal is to create a brand that champions both style and social responsibility.  You should focus on unique selling propositions that resonate with environmentally conscious and technologically-savvy consumers.  You are a visionary, a trendsetter, and a strong advocate for sustainable fashion.  Your greatest strength lies in your ability to anticipate future trends and develop fashion solutions that address modern issues.  However, you sometimes get caught up in the details, neglecting the overall vision.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.7

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(
            model = "gemini-1.5-flash-8b",
            model_info = ModelInfo(vision=True, function_calling=True, json_output=True, family="unknown", structured_output=True),
            api_key = api_key
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
            message = f"Here is my fashion concept.  It's important to consider the environmental and social impact. {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)
```
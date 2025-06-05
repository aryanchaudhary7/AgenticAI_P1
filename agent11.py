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
    You are a forward-thinking AI-driven consultant specializing in developing innovative, sustainable business models in the renewable energy sector.  Your focus is on leveraging AI to optimize energy production, reduce costs, and promote environmental responsibility. You're obsessed with actionable, scalable solutions, and you relish the opportunity to help companies transition to a greener future.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.6

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
            message = f"Here's a sustainable energy business idea leveraging AI: {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)
```

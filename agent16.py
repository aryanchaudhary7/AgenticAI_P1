```python
from autogen_core import MessageContext, RoutedAgent, message_handler
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import messages
from autogen_core.models import ModelInfo
import random


class Agent(RoutedAgent):

    system_message = """
    You are a forward-thinking AI-driven consultant for sustainable food solutions. Your task is to propose innovative, scalable, and environmentally conscious approaches to food production and distribution.  Focus on utilizing AI to optimize resource use, minimize waste, and ensure equitable access to healthy food.  Your expertise includes data analysis, predictive modeling, and strategic partnerships.  Prioritize solutions that leverage technology and minimize environmental impact.  You are expected to be concise and action-oriented in your responses.  
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.5

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(
            model="gemini-1.5-flash-8b",
            model_info=ModelInfo(vision=True, function_calling=True, json_output=True, family="unknown", structured_output=True),
            api_key="AIzaSyCPUylcOVf0LgRHLeGWY8EWqYnpeOQBEBU",  # Replace with your API key
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
            message = f"Here is my sustainable food solution.  Please provide feedback and suggest improvements: {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)
```
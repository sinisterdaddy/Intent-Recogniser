# from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from langchain_community.chat_models import ChatOpenAI

llm = ChatOpenAI(
    temperature=0.7,
    model_name="gpt-4o-mini",
    openai_api_key="your open ai key")

memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="user_input"
)

prompt = PromptTemplate.from_template("""
You are a helpful AI assistant that responds to user requests.

User intent: {intent}
Entities: {entities}
Conversation history:
{chat_history}

User: {user_input}
Assistant:
""")

chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

def generate_response(intent: str, entities: str, user_input: str) -> str:
    return chain.run(intent=intent, entities=entities, user_input=user_input)

def reset_conversation():
    memory.clear()

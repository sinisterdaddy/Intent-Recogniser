from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI

llm = ChatOpenAI(
    temperature=0.7,
    model_name="gpt-4o-mini",
    openai_api_key="your open ai key"
)

prompt = PromptTemplate.from_template("""
Extract useful entities like location, date, destination from this message:
"{text}"

Return as JSON.
""")

def extract_entities(text: str) -> str:
    return llm.invoke(prompt.format(text=text))

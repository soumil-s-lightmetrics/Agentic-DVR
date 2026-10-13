"""LLM shared by the extract and planner nodes."""
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="gpt-5.4-mini",
    temperature=0,
    api_key=os.environ.get("OPENAI_API_KEY")
)

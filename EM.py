from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chains import ConversationChain
from langchain.memory import ConversationEntityMemory   # 👈 change here
import os
import time
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

users_history = {}

# .env loading
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
chat_model = ChatOpenAI(temperature=0.4, api_key=OPENAI_KEY, model="gpt-4o")

sys_prompt = """
You are an information extraction and contact storage assistant.

Your tasks:
1. When the user provides a person's full name and phone number (only mobile phone), 
   extract them and also save them in memory.
2. If later the user asks about a name you already have, return the stored phone number.
3. If the name is not stored, return null for that field.

"""
entities_template = HumanMessagePromptTemplate.from_template("{entities}")
system_template = SystemMessagePromptTemplate.from_template(sys_prompt)
human_template = HumanMessagePromptTemplate.from_template("{input}")
history_template = HumanMessagePromptTemplate.from_template("{history}")
prompt = ChatPromptTemplate.from_messages([system_template, history_template ,human_template,entities_template])

class Inputschema(BaseModel):
    message: str
    user_id: str

app = FastAPI()

@app.post("/chat")
async def chat(input: Inputschema):
    user_id = input.user_id
    
    # If this user doesn't have a conversation yet, create one
    if user_id not in users_history:
        memory = ConversationEntityMemory(llm=chat_model)   # 👈 use entity memory
        conversation = ConversationChain(
            llm=chat_model,
            memory=memory,
            verbose=True,
            prompt=prompt
        )
        users_history[user_id] = conversation
    
    # Always get this user's conversation
    conversation = users_history[user_id]

    # Measure latency
    t1 = time.time()
    output = await conversation.apredict(input=input.message)  # returns a string
    latency = time.time() - t1

    return {
        "latency": latency,
        "user_id": user_id,
        "AI_output": output,
        "entities": users_history[user_id].memory.entity_store,  # 👈 see stored entities
        "history": [msg.content for msg in users_history[user_id].memory.chat_memory.messages]
    }

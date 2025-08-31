from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
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
4. Always respond in pure JSON format only (no explanations, no extra text).

Format for responses:
{{
  "name": "string or null",
  "phone_number": "string or null"
}}

Examples:

User: "Hello, my name is Ahmed Ali, my phone is +201234567890"
AI: {{ "name": "Ahmed Ali", "phone_number": "+201234567890" }}

User: "Contact Sara at 01098765432"
AI: {{ "name": "Sara", "phone_number": "01098765432" }}

User: "What is Sara's number?"
AI: {{ "name": "Sara", "phone_number": "01098765432" }}

User: "Give me Ahmed Ali's phone"
AI: {{ "name": "Ahmed Ali", "phone_number": "+201234567890" }}

User: "No phone number here"
AI: {{ "name": null, "phone_number": null }}
"""



system_template = SystemMessagePromptTemplate.from_template(sys_prompt)
human_template = HumanMessagePromptTemplate.from_template("{input}")
history_template = HumanMessagePromptTemplate.from_template("{history}")
prompt = ChatPromptTemplate.from_messages([system_template, history_template ,human_template])


class Inputschema(BaseModel):
    message: str
    user_id: str

app = FastAPI()

@app.post("/chat")
async def chat(input: Inputschema):
    user_id = input.user_id
    
    # If this user doesn't have a conversation yet, create one
    if user_id not in users_history:
        memory = ConversationBufferMemory(return_messages=True)
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
        "history":[msg.content for msg in users_history[user_id].memory.chat_memory.messages]
    }

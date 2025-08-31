# fastapi_app.py

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chains import ConversationChain
from langchain.memory import ConversationEntityMemory
import os
import time
from langchain.prompts.prompt import PromptTemplate # 👈 Import PromptTemplate

users_history = {}

# .env loading
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
chat_model = ChatOpenAI(temperature=0.4, api_key=OPENAI_KEY, model="gpt-4o")

# Your system prompt remains the same
sys_prompt = """
You are an information extraction and contact storage assistant.

Your tasks:
1. When the user provides a person's full name and phone number (only mobile phone), 
   extract them and also save them in memory.
2. If later the user asks about a name you already have, return the stored phone number.
3. If the name is not stored, return null for that field.

The following is a friendly conversation between a human and an AI. The AI is
talkative and provides lots of specific details from its context. If the AI does not
know the answer to a question, it truthfully says it does not know.
"""

class Inputschema(BaseModel):
    message: str
    user_id: str

app = FastAPI()

@app.post("/chat")
async def chat(input: Inputschema):
    user_id = input.user_id
    
    # If this user doesn't have a conversation yet, create one
    if user_id not in users_history:
        # 1. Initialize memory first
        memory = ConversationEntityMemory(llm=chat_model)
        
        # 2. Get the default template and add your instructions
        # The default template is stored in memory.prompt.template
        _DEFAULT_TEMPLATE = sys_prompt + """

Relevant Information:
{entities}

Conversation:
{history}
Human: {input}
AI:"""
        
        # 3. Create a new, correct PromptTemplate
        prompt = PromptTemplate(
            input_variables=["entities", "history", "input"], template=_DEFAULT_TEMPLATE
        )

        # 4. Pass the correctly formatted prompt to the chain
        conversation = ConversationChain(
            llm=chat_model,
            memory=memory,
            verbose=True,
            prompt=prompt  # 👈 Use the new, correctly structured prompt
        )
        users_history[user_id] = conversation
    
    conversation = users_history[user_id]

    t1 = time.time()
    output = await conversation.apredict(input=input.message)
    latency = time.time() - t1

    return {
        "latency": latency,
        "user_id": user_id,
        "AI_output": output,
        "entities": users_history[user_id].memory.entity_store.dict(), # 👈 Use .dict() for cleaner output
        "history": [msg.content for msg in users_history[user_id].memory.chat_memory.messages]
    }
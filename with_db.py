from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import sqlite3
import json
import os
from dotenv import load_dotenv

# =======================
# 1. Load environment vars
# =======================
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# =======================
# 2. Database setup
# =======================
def init_db():
    """
    Creates a database file 'contacts.db' if not exists,
    and makes sure the 'contacts' table is created.
    """
    conn = sqlite3.connect("contacts.db")  # connect (creates file if not exist)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS contacts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        name TEXT,
        phone_number TEXT
    )
    """)
    conn.commit()
    conn.close()

def save_contact(user_id, name, phone_number):
    """
    Insert a new contact into DB.
    """
    conn = sqlite3.connect("contacts.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO contacts (user_id, name, phone_number) VALUES (?, ?, ?)",
        (user_id, name, phone_number)
    )
    conn.commit()
    conn.close()

def search_contact(user_id, query_name):
    """
    Search contacts by name for a given user_id.
    Using LIKE so it matches even if you give part of the name.
    """
    conn = sqlite3.connect("contacts.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name, phone_number FROM contacts WHERE user_id=? AND name LIKE ?", 
        (user_id, f"%{query_name}%")
    )
    result = cursor.fetchall()
    conn.close()
    return result

# Initialize database at startup
init_db()

# =======================
# 3. LangChain model setup
# =======================
chat_model = ChatOpenAI(temperature=0, api_key=OPENAI_KEY, model="gpt-4o")

sys_prompt = """
You are an information extraction assistant.

Your task:
- Extract the person's full name and phone number (only mobile phone).
- If the current user message does not explicitly include them, use the conversation history to recall the most recently mentioned values, phone and name.

Rules:
- Always return the result in JSON format only.
- If no name or phone number is found in the message or history, return null for that field.

Format:
{{
  "name": "string or null",
  "phone_number": "string or null"
}}
"""




prompt = ChatPromptTemplate.from_messages([
    ("system", sys_prompt),
    ("human", "{input}")
])

# =======================
# 4. FastAPI App
# =======================
app = FastAPI()

class Inputschema(BaseModel):
    message: str
    user_id: str

@app.post("/chat")
async def chat(input: Inputschema):
    """
    Main chat endpoint:
    1. If user asks for a phone number, search DB.
    2. Otherwise, extract info using AI and save to DB if found.
    """
    # Step 1: Check if the user is asking for stored contact
    if "number" in input.message.lower() or "phone" in input.message.lower():
        results = search_contact(input.user_id, input.message)
        if results:
            return {
                "contacts": [{"name": r[0], "phone_number": r[1]} for r in results]
            }

    # Step 2: Otherwise, ask AI to extract data
    chain = prompt | chat_model
    output = await chain.ainvoke({"input": input.message})

    # Step 3: Parse AI output into JSON
    try:
        extracted = json.loads(output.content)
    except:
        extracted = {"name": None, "phone_number": None}

    # Step 4: If contact info found, save it
    if extracted["name"] and extracted["phone_number"]:
        save_contact(input.user_id, extracted["name"], extracted["phone_number"])

    return extracted

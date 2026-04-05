import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")

def generate_llm_answer(question: str, context: list[str]) -> str:
    prompt = f"""
Use ONLY the following document context to answer the question.

Context:
{chr(10).join(context)}

Question:
{question}

Rules:
- Do not use outside knowledge
- Cite context if relevant
"""

    response = model.generate_content(prompt)
    return response.text

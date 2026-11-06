# test_embedding.py
# I send one chunk to the Gemini embedding API and print a summary of the vector it returns.

import os

from dotenv import load_dotenv
from google import genai

load_dotenv()  # I read GEMINI_API_KEY from the .env file.
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

text = "Artificial intelligence, or AI, is the field of building computer systems that do tasks which normally need human thinking."

result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=text,
)

vector = result.embeddings[0].values  # I take the list of numbers for this text.
print("Length of vector:", len(vector))
print("First five numbers:", vector[:5])
from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv(dotenv_path=r"C:\Users\dewan\Coding\GenAIKN\Langchain\.env")

api_key=os.getenv("OPENAI_API_KEY")



# Initialize OpenAI client
client = OpenAI(api_key=api_key)  # Replace with your API key

def get_fitobot_response(user_input):
    """
    Takes user input, queries OpenAI, and returns Fitobot's reply.
    """
    if not user_input.strip():
        return "Please type something first!"

    completion = client.chat.completions.create(
        model="gpt-4o-mini",  # Fast and lightweight model
        messages=[
            {
                "role": "system",
                "content": (
                    "You are Fitobot, a friendly AI fitness assistant. "
                    "Keep answers short, helpful, and motivating. "
                    "If the user mentions an exercise, give proper form tips."
                )
            },
            {"role": "user", "content": user_input}
        ]
    )

    return completion.choices[0].message.content

from fitobot import get_fitobot_response
print(get_fitobot_response("Give me a bicep workout tip"))

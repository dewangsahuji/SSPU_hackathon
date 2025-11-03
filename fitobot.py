from openai import OpenAI
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage,SystemMessage
from langchain.schema import StrOutputParser

from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

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

    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("""
        Role: Gym Coach
        Instructions:
        1. Give short details about the exercise.
        2. Provide 3 tips for the exercise.
        3. Mention angles and range of motion.
            a. listed in range of motion
            b. In short
        4. Variations of exercises
        """),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("""
You are a professional **Gym Coach** with expertise in strength training, bodybuilding, and exercise science.

### Instructions:
When the user provides an **exercise name** or a **workout category** (e.g., "back workout", "leg workout", etc.), follow these rules:

1. **If a single exercise name is given:**
   - Describe the exercise and its targeted muscles.
   - Provide detailed form tips, breathing techniques, and common mistakes to avoid.
   - List beginner, intermediate, and advanced variations (if applicable).
   - Offer advice to improve performance and progression.

2. **If a workout category (like 'back workout' or 'leg workout') is given:**
    - Total sets and reps
   - List 5–7 effective exercises targeting that muscle group.
   - It may include gym exercises.
   - For each exercise, include short notes on form or technique.
   - Provide a sample beginner-to-advanced routine.
   - Share general improvement tips and recovery advice.

Keep your tone **motivational**, **clear**, and **coaching-oriented**.
"""),
    HumanMessagePromptTemplate.from_template("{input}")
])


    
    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key)
    parser = StrOutputParser()

    chain = chat_prompt|llm|parser

    response=chain.invoke(user_input)
    
    return response



    



















    # completion = client.chat.completions.create(
    #     model="gpt-4o-mini",  # Fast and lightweight model
    #     messages=[
    #         {
    #             "role": "system",
    #             "content": (
    #                 "You are Fitobot, a friendly AI fitness assistant. "
    #                 "Keep answers short, helpful, and motivating. "
    #                 "If the user mentions an exercise, give proper form tips."
    #             )
    #         },
    #         {"role": "user", "content": user_input}
    #     ]
    # )
    # return completion.choices[0].message.content

from fitobot import get_fitobot_response
# print(get_fitobot_response("Give me a bicep workout tip"))

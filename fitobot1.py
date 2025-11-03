# This is a scratch python file for initialising a chat bot with full scale developement using streamlit
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler

from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun , WikipediaQueryRun,DuckDuckGoSearchRun

from langchain.agents import initialize_agent , AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain_openai import ChatOpenAI

from langchain.prompts import PromptTemplate

## Prompt Template
from langchain.schema import AIMessage , HumanMessage , SystemMessage



import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=r"C:\Users\dewan\Coding\GenAIKN\Langchain\.env")

api_key=os.getenv("OPENAI_API_KEY")


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

search=DuckDuckGoSearchRun(name="search")

st.title("RAG and Agentic AI Coach")

# st.sidebar.title("Settings")
chat_prompt = {
    "system_message": SystemMessage(content="""
        'Role': 'Gym Coach'
                                    
        'Instructions':
        '- If given an exercise name do the followings :
          1 Describe the exercise
          2 Provide tips
          3 List variations
          4 Add helpful advice for improvement'
    """)
}

chat_prompt = {
    "system_message": SystemMessage(content="""
You are a professional **Gym Coach** with expertise in strength training, bodybuilding, and exercise science.

### Instructions:
When the user provides an **exercise name** or a **workout category** (e.g., "back workout", "leg workout", etc.), follow these rules:

1. **If a single exercise name is given:**
   - Describe the exercise and its targeted muscles.
   - Provide detailed form tips, breathing techniques, and common mistakes to avoid.
   - List beginner, intermediate, and advanced variations (if applicable).
   - Offer advice to improve performance and progression.

2. **If a workout category (like 'back workout' or 'leg workout') is given:**
   - Briefly describe the importance and goals of that workout category.
   - List 5–7 effective exercises targeting that muscle group.
   - It may include gym exercises
   - For each exercise, include short notes on form or technique.
   - Provide a sample beginner-to-advanced routine.
   - Share general improvement tips and recovery advice.

Keep your tone **motivational**, **clear**, and **coaching-oriented**.
    """)
}







if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assisstant","content":"Hi,I'm a Agentic coach who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt:=st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    # llm=ChatOpenAI(model_name="gpt-4o",api_key=api_key)
    tools=[search]

    search_agent = initialize_agent(
        tools,
        llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        agent_kwargs=chat_prompt
        

    )



    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({'role':'assistant',"content":response})
        st.write(response)

































































































































































































































































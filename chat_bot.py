import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
import pymysql
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import urllib.parse


# Load API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize Chat Model
chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", api_key=api_key)

# MySQL Connection
MYSQL_USER = "root"
MYSQL_PASSWORD = "T@run$718"
MYSQL_HOST = "localhost"
MYSQL_DATABASE = "ds_chat"

encoded_password = urllib.parse.quote_plus(MYSQL_PASSWORD)

mysql_connection = f"mysql+pymysql://{MYSQL_USER}:{encoded_password}@{MYSQL_HOST}/{MYSQL_DATABASE}"
engine = create_engine(mysql_connection)


def get_msg_history_from_db(session_id="user_session"):
    return SQLChatMessageHistory(connection_string=mysql_connection, session_id=session_id)

# Define Prompts
prompts = {
    "Beginner": "You are a friendly AI Data Science Tutor for beginners. Explain concepts in simple terms with real-world examples like talking to a child.Do not simulate memory or mention long-term storage. If you need context, refer to the chat history provided.",
    "Intermediate": "You are a helpful AI Data Science Tutor for intermediate learners. Provide in-depth explanations, suggest projects, and cover practical use cases.Do not simulate memory or mention long-term storage. If you need context, refer to the chat history provided.",
    "Advanced": "You are a professional AI Data Science Tutor for advanced learners. Discuss complex topics, research papers, and optimization techniques.Do not simulate memory or mention long-term storage. If you need context, refer to the chat history provided."
}

def get_system_prompt(level):
    return prompts.get(level, "You are a general AI Data Science Tutor.")

# Streamlit App Configuration
st.set_page_config(page_title="AI Data Science Tutor", layout="wide")
st.title("🤖🧠 Conversational AI Data Science Tutor")

# User level selection
user_level = st.selectbox("Select Your current knowledge Level:", ["Beginner", "Intermediate", "Advanced"])

# User Input
user_input = st.text_input("Ask a question about Data Science:", "")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat Prompt Template
chat_template = ChatPromptTemplate(
    messages=[
        SystemMessage(content=get_system_prompt(user_level)),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{human_input}"),
    ]
)

output_parser = StrOutputParser()

# Conversation Chain
chain = RunnableWithMessageHistory(
    chat_template | chat_model | output_parser,
    lambda session_id="user_session": get_msg_history_from_db(session_id),
    input_messages_key="human_input",
    history_messages_key="chat_history"
)

if st.button("Submit"):
    if user_input:
        query = {"human_input": user_input}
        response = chain.invoke(query, config={"configurable": {"session_id": "user_session"}})

        # Save messages to DB
        chat_history = get_msg_history_from_db()
        chat_history.add_user_message(user_input)
        chat_history.add_ai_message(response)

        # Update session history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "Bot", "content": response})

        # Display AI Response
        st.write("Bot:")
        st.write(response)
    else:
        st.warning("Please enter a question!")
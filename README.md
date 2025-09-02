# mental health chatbot
# Developed an NLP-based mental health chatbot using Python and Streamlit to provide conversational support and curated resources. Designed to answer basic queries, recommend coping strategies, and enhance user accessibility to mental health information

!pip install streamlit pyngrok --quiet

code = '''
import zipfile
import os
import json
import random
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import io

# Unzip and Load Data
zip_path = "archive (4).zip"
extract_to = "mental_health_data"

if not os.path.exists(extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Load JSON
json_path = os.path.join(extract_to, "intents.json")
with open(json_path, "r") as file:
    data = json.load(file)

# Prepare Training Data
patterns = []
labels = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        labels.append(intent['tag'])

# Convert to DataFrame for analysis
df = pd.DataFrame({'Pattern': patterns, 'Tag': labels})

# Sidebar for Data Exploration
st.sidebar.title("📊 Dataset Exploration")

if st.sidebar.checkbox("Show Data Info"):
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.sidebar.text(s)

if st.sidebar.checkbox("Show Head"):
    st.sidebar.dataframe(df.head())

if st.sidebar.checkbox("Show Tail"):
    st.sidebar.dataframe(df.tail())

if st.sidebar.checkbox("Show Dataset Size"):
    st.sidebar.write(f"Total rows: {df.shape[0]}")
    st.sidebar.write(f"Total columns: {df.shape[1]}")

if st.sidebar.checkbox("Show Unique Tags Count"):
    st.sidebar.write(f"Unique Tags: {df['Tag'].nunique()}")

if st.sidebar.checkbox("Show All Unique Tags"):
    st.sidebar.write(df['Tag'].unique())

if st.sidebar.checkbox("Show Message Count Per Tag"):
    st.sidebar.dataframe(df.groupby("Tag").size().reset_index(name='Count'))

selected_tag = st.sidebar.selectbox("Select a tag to view messages", options=sorted(df['Tag'].unique()))
if st.sidebar.button("Show Messages for Selected Tag"):
    st.sidebar.dataframe(df[df['Tag'] == selected_tag])

if st.sidebar.checkbox("Show Tag Distribution Chart"):
    fig, ax = plt.subplots()
    df['Tag'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title("Intent Tag Distribution")
    ax.set_xlabel("Tags")
    ax.set_ylabel("Count")
    st.sidebar.pyplot(fig)

# Train Model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(patterns)
y = np.array(labels)

model = MultinomialNB()
model.fit(X, y)

# Streamlit UI
st.title("💬 Mental Health Chatbot")

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "show_history" not in st.session_state:
    st.session_state.show_history = True
if "exit_chat" not in st.session_state:
    st.session_state.exit_chat = False

# Buttons for control
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_history = []
with col2:
    if st.button("👁 Toggle Chat History"):
        st.session_state.show_history = not st.session_state.show_history
with col3:
    if st.button("🚪 Exit Chatbot"):
        st.session_state.exit_chat = True
        st.session_state.chat_history.append(("System", "You have exited the chat. See you next time!"))

# Exit Mode
if st.session_state.exit_chat:
    st.markdown("**⚠️ Chatbot session ended. Refresh to restart.**")
    st.stop()

# Chat input
user_input = st.text_input("You:", key="input")

if user_input:
    X_input = vectorizer.transform([user_input])
    predicted_tag = model.predict(X_input)[0]

    response = "I'm here for you."
    for intent in data['intents']:
        if intent['tag'] == predicted_tag:
            response = random.choice(intent['responses'])
            break

    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Chatbot", response))

# Display chat history if visible
if st.session_state.show_history:
    for sender, message in st.session_state.chat_history:
        if sender == "Chatbot":
            st.markdown(f"**🤖 {sender}:** {message}")
        elif sender == "System":
            st.markdown(f"**⚠️ {sender}:** {message}")
        else:
            st.markdown(f"**🧑 {sender}:** {message}")

'''

with open('app.py', 'w') as f:
    f.write(code)

from google.colab import files
uploaded = files.upload()

import os
from pyngrok import ngrok

# Set ngrok token here
os.environ["NGROK_AUTHTOKEN"] = "2yMcwrhudEQfRg0NSaQUZXqFpz9_4aMzfjqS2S8bEQb4PTC5T"
ngrok.set_auth_token(os.environ["NGROK_AUTHTOKEN"])

# Launch ngrok tunnel
public_url = ngrok.connect(8501)
print(f"🔗 Your app is live at: {public_url}")

# Run the app
!streamlit run app.py &

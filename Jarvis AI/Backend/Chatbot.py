from groq import Groq
from json import load, dump
import datetime
from dotenv import dotenv_values
import cohere


# Load environment variables and initialize Cohere client
env_vars = dotenv_values(".env")


# CohereAPIKEY = env_vars.get("CohereAPIKey")

# co = cohere.Client(api_key=CohereAPIKEY)
Username = env_vars.get("Username")
Assistantname = env_vars.get("Assistantname")
GroqAPIKey = env_vars.get("GroqAPIKey")

# Initialize Groq API client
client = Groq(api_key=GroqAPIKey)

# Chat message history
messages = []

# System prompt
System = f"""Hello, I am {Username}, You are a very accurate and advanced AI chatbot named {Assistantname} which also has real-time up-to-date information from the internet.
*** Do not tell time until I ask, do not talk too much, just answer the question.***
*** Reply in only English, even if the question is in Hindi, reply in English.***
*** Do not provide hiinotes in the output, just answer the question and never mention your training data. ***
"""

# Fix: Correct system message format
SystemChatBot = [
    {"role": "system", "content": System}
]
 
# Load chat log from JSON (if exists)
try:
    with open(r"Data\ChatLog.json", "r") as f:
        messages = load(f)
except FileNotFoundError:
    with open(r"Data\ChatLog.json", "w") as f:
        dump([], f)

# Function to get real-time information
def RealtimeInformation():
    current_date_time = datetime.datetime.now()
    day = current_date_time.strftime("%A")
    date = current_date_time.strftime("%d")
    month = current_date_time.strftime("%B")
    year = current_date_time.strftime("%Y")
    hour = current_date_time.strftime("%H")
    minute = current_date_time.strftime("%M") 
    second = current_date_time.strftime("%S")

    data = f"Pleasse use this real-time information if needed,\n"
    data += f"Day: {day}\n Date: {date}\n Month: {month}\n Year:{year}\n"
    data += f"Time : {hour} hours:{minute}minutes : {second}.\n"
    return data

# Function to clean chatbot response
def AnswerModifier(Answer):
    lines = Answer.split("\n")
    non_empty_lines = [line  for line in lines if line.strip()]
    modified_answer ='\n'.join(non_empty_lines)
    return modified_answer

# Chatbot function
def ChatBot(Query):
    try:
        # Load existing messages
        with open(r"Data\ChatLog.json", "r") as f:
            messages = load(f)

        # Append user query
        messages.append({"role": "user", "content":f"{Query}"})

        # Fix: Ensure correct role and structure
        # chat_payload = SystemChatBot + [{"role": "system", "content": RealtimeInformation()}] + messages

        # API Call
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=SystemChatBot + [{"role":"system","content":RealtimeInformation()}] + messages,
            max_tokens=1024,
            temperature=0.7,
            top_p=1,
            stream=True,
            stop=None
        )

        Answer = ""

        for chunk in completion:
            if chunk.choices[0].delta.content:
                Answer += chunk.choices[0].delta.content

        # Remove unwanted tokens
        Answer = Answer.replace("</s>", "")

        # Save chatbot response
        messages.append({"role": "assistant", "content": Answer})

        # Save updated chat log
        with open(r"Data\ChatLog.json", "w") as f:
            dump(messages, f, indent=4)

        return AnswerModifier(Answer=Answer)

    except Exception as e:
        print(f"Error: {e}")
        with open(r"Data\ChatLog.json","w") as f:
            dump([],f,indent=4)
        return ChatBot(Query)

# Main loop
if __name__ == "__main__":
    while True:
        user_input = input("Enter Your Query: ")
        print(ChatBot(user_input))

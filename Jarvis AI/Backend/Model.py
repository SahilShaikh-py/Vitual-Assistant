import cohere 
from rich import print
from dotenv import dotenv_values

# Load environment variables from the .env file
env_vars = dotenv_values(".env")

# Retrieve the Cohere API key
CohereAPIKey = env_vars.get("CohereAPIKey")  # Ensure the key is correctly referenced

# Debugging line to check if the API key is loaded
co = cohere.Client(api_key=CohereAPIKey)  # Create a Cohere client

# Define recognized function keywords for task performance
funcs = ["exit", "general", "realtime", "open", "close",
         "play", "generate image", "system", "content",
         "google search", "youtube search", "reminder", "virtual Mouse"]

# Store user messages
messages = []

# Define preamble to guide the AI model
preamble = """
You are a very accurate Decision-Making Model, which decides what kind of a query is given to you.
You will decide whether a query is a 'general' query, a 'realtime' query, or is asking to perform any task or automation like 'open facebook, instagram', 'can you write an application and open it in notepad'
*** Do not answer any query, just decide what kind of query is given to you. ***
...
"""

ChatHistory = [
    {"role": "user", "message": "How are You Sir "},
    {"role": "chatbot", "message": "general How are You?"},
    {"role": "user", "message": "Do you like Biryani"},
    {"role": "chatbot", "message":"general do you like Biryani?"},
    {"role": "user", "message": "open chrome and tell me about chatrapati shivaji maharaj"},
    {"role": "chatbot", "message": "open chrome,general tell me about chatrapati shivaji Maharaj."},
    {"role": "user", "message": "open chrome and firefox"},
    {"role": "chatbot", "message": "open chrome,open firefox"},
    {"role": "user", "message": "what is today's date and by the way remind me that i have exam on 5th april"},
    {"role": "chatbot", "message": "general what is today's date ,reminder i have exam on 5th april"},
    {"role": "user", "message": "chat with me."},
    {"role": "chatbot", "message": "general chat with me"},
]

def FirstLayerDMM(prompt: str = "test"):
    messages.append({"role": "user", "content": f"{prompt}"})
    
    stream = co.chat_stream(
        model='command-r-plus',
        message=prompt,
        temperature=0.7,
        chat_history=ChatHistory,
        prompt_truncation='OFF',
        connectors=[],
        preamble=preamble
    )

    response = ""

    for event in stream:
        if event.event_type == "text-generation":
            response += event.text

    response = response.replace("\n", "")
    response = response.split(",")

    response = [i.strip() for i in response]

    temp = []

    for task in response:
        for func in funcs:
            if task.startswith(func):
                temp.append(task)
    response = temp

    if "(query)" in response:
        newresponse = FirstLayerDMM(prompt=prompt)
        return newresponse
    else:
        return response
    
if __name__ == "__main__":
    while True:
        print(FirstLayerDMM(input(">>>  ")))

pip install openai==1.0.0

pip install load_dotenv
import openai
import os

from dotenv import load_dotenv
OPENAI_API_KEY = "API-KEY"

# Load the environment variables
load_dotenv()

# Set the OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

%env OPENAI_API_KEY=API-KEY

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0, 
    )
    return response.choices[0].message.content
prompt = f"""
Your task is to direct and give instructions to do's and dont's when a patient has arthritis

Write a description based on the information
provided only by not providing extra information in the technical specifications delimited by
triple backticks.

Use at most 100 words.

"""
response = get_completion(prompt)
print(response)
import openai
import os
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()


openai.api_key = os.getenv('OPENAI_API_KEY')

predicted_disease = ""

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message.content
def predict_diseases(symptoms):
    # Combine symptoms into a single prompt
    prompt = "I am experiencing " + ", ".join(symptoms) + ". What could it be?"

    # Get the model's prediction
    global predicted_disease
    predicted_disease = get_completion(prompt)
    return predicted_disease

def chat_with_bot():
    print("Bot: Hello! I'm here to assist you. Let's get started.")

    # Ask predefined questions
    name = input("Bot: May I know your name? ")
    age = input("Bot: How old are you, {}? ".format(name))
    issue = input("Bot: How can I help you today? ")

    # Initialize a list to store symptoms
    symptoms = []

    # Ask if there are any additional symptoms
    additional_symptoms = input("Bot: Do you have any other symptoms you'd like to share? If not, you can type 'none'. ")

    # Store symptoms until the user indicates none
    while additional_symptoms.lower() != "done":
        symptoms.append(additional_symptoms)
        additional_symptoms = input("Bot: Any other symptoms? If not, you can type 'done'. ")

    if symptoms:
        print("Bot: Let me analyze your symptoms...")
        prediction = predict_diseases(symptoms)
        print("Bot:", prediction)

    user_input = input("You: ")
    while user_input.lower() != "exit":
        # Send user input to the chatbot and get response
        # bot_response = get_completion(user_input)
        # print("Bot:", bot_response)

        if predicted_disease:
            question_with_disease = user_input + predicted_disease
            disease_response = get_completion(question_with_disease)
            print("Bot:", disease_response)

        user_input = input("You: ")


chat_with_bot()

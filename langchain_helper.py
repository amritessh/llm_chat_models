from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# print(llm.invoke("What is the capital of France?"))

def generate_response(animal_type):
    # Get configuration from environment variables
    model_name = os.getenv('MODEL_NAME', 'gemini-2.0-flash')
    temperature = float(os.getenv('TEMPERATURE', '0.7'))
    max_tokens = int(os.getenv('MAX_OUTPUT_TOKENS', '1024'))

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=temperature,
        max_output_tokens=max_tokens
    )

    prompt_template_name = ChatPromptTemplate.from_messages([
        ("human", "Give me 5 creative names for my pet {animal_type}. Format the response as a numbered list.")
    ])

    # Create a chain using the new pipe syntax
    chain = prompt_template_name | llm | StrOutputParser()

    # Get and format the response
    response = chain.invoke({"animal_type": animal_type})
    
    # Print with some formatting
    print("\n=== Pet Name Suggestions ===")
    print(f"For your {animal_type}:")
    print(response)
    print("===========================\n")

    if os.getenv('DEBUG_MODE', 'False').lower() == 'true':
        logging.debug(f"Generated response for {animal_type}: {response}")
        
    return response

if __name__ == "__main__":
    generate_response(input("Enter an animal type: "))




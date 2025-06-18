from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.google_search import GoogleSearchRun
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
from dotenv import load_dotenv
import os
import logging
import re

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def check_environment():
    """Check if required environment variables are set."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found in environment variables")
        print("Please set it in your .env file")
        return False
    print("✅ Environment variables loaded successfully")
    return True

def generate_response(animal_type, pet_color):
    """Generate creative pet names using LangChain and Gemini."""
    try:
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
            ("human", "Give me 5 creative names for my {pet_color} {animal_type}. Format the response as a numbered list.")
        ])

        # Create a chain using the new pipe syntax
        chain = prompt_template_name | llm | StrOutputParser()

        # Get and format the response
        response = chain.invoke({"animal_type": animal_type, "pet_color": pet_color})
        
        # Print with some formatting
        print("\n=== Pet Name Suggestions ===")
        print(f"For your {pet_color} {animal_type}:")
        print(response)
        print("===========================\n")

        if os.getenv('DEBUG_MODE', 'False').lower() == 'true':
            logging.debug(f"Generated response for {pet_color} {animal_type}: {response}")
            
        return response
    
    except Exception as e:
        logging.error(f"Error generating pet names: {e}")
        print(f"❌ Error generating pet names: {e}")
        return None

def calculator_tool(expression):
    """Simple calculator tool for mathematical operations."""
    try:
        # Clean the expression to only allow safe mathematical operations
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        
        # Use eval with restricted globals for basic math
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error calculating: {str(e)}"

def wikipedia_search_tool(query):
    """Search Wikipedia for information."""
    try:
        wikipedia = WikipediaAPIWrapper()
        result = wikipedia.run(query)
        # Limit result length to avoid token limits
        return result[:2000] + "..." if len(result) > 2000 else result
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"

def manual_agent_with_tools():
    """Manual implementation of an agent that can use tools."""
    try:
        print("🤖 Initializing manual agent with tools...")
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.7,
            max_output_tokens=1024
        )
        
        # Define the tools
        tools = {
            "wikipedia": wikipedia_search_tool,
            "calculator": calculator_tool
        }
        
        # Create the agent prompt
        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant that can use tools to answer questions.

Available tools:
- wikipedia: Search Wikipedia for information. Usage: wikipedia("search query")
- calculator: Perform mathematical calculations. Usage: calculator("mathematical expression")

When you need to use a tool, respond with the tool name and input in this exact format:
TOOL: tool_name
INPUT: tool_input

After seeing the tool result, provide your final answer.

If you don't need any tools, just answer the question directly."""),
            ("human", "{input}")
        ])
        
        chain = agent_prompt | llm | StrOutputParser()
        
        query = "What is the average age of a dog? Multiply the average age by 10 and tell me the result."
        print(f"🚀 Processing query: {query}")
        
        max_iterations = 5
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")
            
            response = chain.invoke({"input": query})
            print(f"Agent response: {response}")
            
            # Check if the agent wants to use a tool
            if "TOOL:" in response and "INPUT:" in response:
                # Extract tool name and input
                tool_match = re.search(r'TOOL:\s*(\w+)', response)
                input_match = re.search(r'INPUT:\s*(.+)', response)
                
                if tool_match and input_match:
                    tool_name = tool_match.group(1).lower()
                    tool_input = input_match.group(1).strip()
                    
                    print(f"🔧 Using tool: {tool_name}")
                    print(f"📝 Tool input: {tool_input}")
                    
                    if tool_name in tools:
                        tool_result = tools[tool_name](tool_input)
                        print(f"🔍 Tool result: {tool_result}")
                        
                        # Update the query to include the tool result
                        query = f"Previous query: {query}\nTool used: {tool_name}\nTool result: {tool_result}\nNow provide the final answer."
                    else:
                        print(f"❌ Unknown tool: {tool_name}")
                        break
                else:
                    print("❌ Could not parse tool usage")
                    break
            else:
                # No tool usage detected, this should be the final answer
                print("\n=== Final Agent Result ===")
                print(response)
                print("========================\n")
                return response
        
        print("❌ Maximum iterations reached")
        return "Sorry, I couldn't complete the task within the maximum number of iterations."
    
    except Exception as e:
        logging.error(f"Error running manual agent: {e}")
        print(f"❌ Error running manual agent: {e}")
        return None

def simple_chain_solution():
    """A simple chain-based solution without agents."""
    try:
        print("🔗 Running simple chain solution...")
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.1,  # Lower temperature for more consistent results
            max_output_tokens=1024
        )
        
        # First, get information about dog lifespan
        dog_info_prompt = ChatPromptTemplate.from_messages([
            ("human", "What is the average lifespan/age of a dog? Please provide just the number in years.")
        ])
        
        dog_chain = dog_info_prompt | llm | StrOutputParser()
        dog_age_response = dog_chain.invoke({})
        
        print(f"Dog age response: {dog_age_response}")
        
        # Extract the number from the response
        import re
        numbers = re.findall(r'\d+', dog_age_response)
        if numbers:
            average_age = int(numbers[0])
            result = average_age * 10
            
            final_answer = f"The average age of a dog is approximately {average_age} years. Multiplying {average_age} by 10 gives us {result}."
            
            print("\n=== Simple Chain Result ===")
            print(final_answer)
            print("==========================\n")
            
            return final_answer
        else:
            print("❌ Could not extract age from response")
            return None
    
    except Exception as e:
        logging.error(f"Error running simple chain: {e}")
        print(f"❌ Error running simple chain: {e}")
        return None

if __name__ == "__main__":
    if not check_environment():
        exit(1)
    
    print("Trying different approaches to solve the agent problem...\n")
    
    print("=== Approach 1: Simple Chain Solution ===")
    simple_chain_solution()
    
    print("\n=== Approach 2: Manual Agent with Tools ===")
    manual_agent_with_tools()
    
    print("\n=== Approach 3: Pet Name Generator ===")
    # Uncomment to test pet name generator
    # generate_response("cat", "orange")
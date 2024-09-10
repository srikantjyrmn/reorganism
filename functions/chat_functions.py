import os
from firebase_utils import download_directory, vector_search
from prompts import self_rag_agent
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import glob


def chat_with_openai(message, system_prompt = self_rag_agent):
    """Chat with the OpenAI API
    """
    # Get the API key from environment variable
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("OPENAI_API_KEY not found in environment variables")
        return "OPENAI_API_KEY not found in environment variables"

    print(f"API Key found: {api_key[:5]}...") # Print first 5 characters for verification

    # Set the OpenAI API key
    #openai.api_key = api_key

    try:
        print("Sending request to OpenAI API...")
        response = OpenAI(api_key = api_key).chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            max_tokens=1000,
            temperature=0.5
        )
        print("Successfully received response from OpenAI API")
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in OpenAI API request: {e}")
        return f"Error in OpenAI API request: {e}"

def chat_with_Claude(message):
    # Get the API key from environment variable
    api_key = os.environ.get('CLAUDE_API_KEY')
    if not api_key:
        print("CLAUDE_API_KEY not found in environment variables")
        return "CLAUDE_API_KEY not found in environment variables"

    print(f"API Key found: {api_key[:5]}...") # Print first 5 characters for verification

    # Create an instance of the Anthropics client
    client = anthropic.Anthropic(api_key=api_key)

    try:
        print("Sending request to Claude API...")
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1000,
            temperature=0,
            system="You are a world-class poet. Respond only with short poems.",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": message
                        }
                    ]
                }
            ]
        )
        print("Successfully received response from Claude API")
        return response.content
    except Exception as e:
        print(f"Error in Claude API request: {e}")
        return f"Error in Claude API request: {e}"

def load_index():
    index_name = 'BlogPosts'
    index_path = f'.ragatouille/colbert/indexes/{index_name}'
    local_path = f'local_indices/{index_path}'
    files = glob.glob(local_path)

    if len(files) == 0:
        # Run something if there are no files
        print("No files found at the given location. Downloading")
        download_directory(index_path, local_path)
    else:
        # Continue with your code if files are found
        print("Files found at the given location:", files)
    return local_path

def chat_with_index(message):
    print("loading index")
    retrieved_context = vector_search(message, k = 20)
    print("searching index")
    
    print(f"got context {retrieved_context}")
    context = [{'document_id': obj['document_id'], 'content': obj['contents'], 'passage_id': obj['chunk_id']} for obj in retrieved_context]
    response = chat_with_openai(
        system_prompt = self_rag_agent.format(
            context = context), 
        message = message
    )
    return response


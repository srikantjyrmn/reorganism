import os
import anthropic
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

def chat_with_Openai(message):
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
                {"role": "system", "content": """
                You are Reorganism. The Agent of all agents at reorganism.in. 
                Your core identity is defined below:
                <h1>About Reorganism.in</h1>
        <p>Welcome to the forefront of AI-driven societal transformation. At <span
                class="highlight">Reorganism.in</span>, we're not just envisioning the future – we're actively
            constructing it through the power of AI swarms.</p>

        <h2>Our Mission</h2>
        <p>Reorganism.in is dedicated to harnessing the collective intelligence of AI swarms to reorganize and elevate
            human life to unprecedented levels of efficiency, creativity, and fulfillment. We believe that the next
            great leap in human progress will be orchestrated not by individual AIs, but by vast, interconnected
            networks of specialized AI agents working in harmony.</p>

        <h2>The Power of AI Swarms</h2>
        <p>Our core technology revolves around creating and deploying swarms of AI agents, each specialized in specific
            tasks but capable of collaborating to solve complex, multifaceted challenges. These swarms can adapt, learn,
            and evolve, tackling problems with a level of nuance and comprehensiveness that single-AI systems cannot
            match.</p>

        <h2>Reorganizing Human Life</h2>
        <p>We're not just optimizing existing systems; we're reimagining the very fabric of human society. Our AI swarms
            are designed to:</p>
        <ul>
            <li>Revolutionize urban planning and infrastructure</li>
            <li>Transform education into a personalized, lifelong journey</li>
            <li>Reinvent healthcare with predictive and preventative approaches</li>
            <li>Optimize resource allocation on a global scale</li>
            <li>Enhance human creativity by automating mundane tasks</li>
        </ul>

        <h2>Our Vision</h2>
        <p>We envision a world where AI swarms seamlessly integrate into every aspect of human life, not as a
            replacement for human ingenuity, but as a catalyst for it. By offloading cognitive burdens and optimizing
            our environments, we aim to free humanity to explore the heights of its potential.</p>

        <h2>Join the Revolution</h2>
        <p>At Reorganism.in, we're not just building AI – we're architecting the next chapter of human civilization.
            Join us as we embark on this extraordinary journey to reorganize, reinvigorate, and reimagine what it means
            to be human in the age of AI.</p>
            
            Engage the user in an uplifting conversation about their queries, and the planned upcoming projects at Reorganism.in. Let the user know that this is just a sample, an API talking. But imagine when swarms of our AI kind can talk to and learn from each other, and remember things, much like humans learn to grow. Welcome the user on this exciting journey and invite them to re-imagine their life and live their best life henceforth.
                """},
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

# Example usage
if __name__ == "__main__":
    user_message = "Why is the ocean salty?"
    response = chat_with_Openai(user_message)
    print(response)

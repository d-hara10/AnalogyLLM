import openai
import torch
import os, json
from pathlib import Path
import datetime
from dotenv import load_dotenv
import logging
import sys

# === SETUP LOGGING ===
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analogy_generator.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# === CONFIG ===
PROJECT_DIR = "/Users/derek/Desktop/CLUBS/LLM Analogy"
ROLLOUTS_DIR = f"{PROJECT_DIR}/rollouts"
NUM_ANALOGIES = 3  # Number of analogies to generate at once

# Load environment variables
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    logger.error("No OpenAI API key found in .env file")
    raise ValueError("OpenAI API key not found in .env file")

logger.info("API Key found (first 4 chars): %s...", api_key[:4])
openai.api_key = api_key

# System message to guide the model
SYSTEM_MESSAGE = """You are an expert at creating mathematical analogies across different domains. 
Your task is to explain mathematical concepts using analogies from various fields like:
- Biology (genes, cells, evolution)
- Physics (forces, energy, waves)
- Chemistry (reactions, bonds, elements)
- Computer Science (algorithms, data structures)
- Nature (ecosystems, weather, geology)
- Society (organizations, networks, growth)

Create clear, accurate, and insightful analogies that help understand mathematical concepts.
Each analogy should be detailed, accurate, and use domain-specific terminology appropriately."""

def generate_analogies(prompt, num_analogies=NUM_ANALOGIES):
    """Generate multiple analogies for a given prompt"""
    analogies = []
    
    # Different generation parameters for each analogy
    generation_params = [
        {
            "temperature": 0.9,  # More creative
            "top_p": 0.9,
            "presence_penalty": 0.6,
            "frequency_penalty": 0.6
        },
        {
            "temperature": 0.7,  # More focused
            "top_p": 0.8,
            "presence_penalty": 0.8,
            "frequency_penalty": 0.8
        },
        {
            "temperature": 1.1,  # Most creative
            "top_p": 0.95,
            "presence_penalty": 1.0,
            "frequency_penalty": 1.0
        }
    ]
    
    for i in range(num_analogies):
        try:
            logger.info(f"Attempting to generate analogy {i+1} with GPT-4")
            logger.debug(f"Using parameters: {generation_params[i]}")
            
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                **generation_params[i]
            )
            analogy = response.choices[0].message.content.strip()
            logger.info(f"Successfully generated analogy {i+1}")
            logger.debug(f"Generated analogy: {analogy[:100]}...")  # Log first 100 chars
            analogies.append(analogy)
            
        except openai.AuthenticationError as e:
            logger.error(f"Authentication Error: {str(e)}")
            logger.error("Please check your API key in the .env file")
            analogies.append("Authentication Error: Please check your API key")
            
        except openai.RateLimitError as e:
            logger.error(f"Rate Limit Error: {str(e)}")
            logger.error("You've hit the API rate limit. Please wait a moment and try again.")
            analogies.append("Rate Limit Error: Please wait and try again")
            
        except openai.APIError as e:
            logger.error(f"API Error: {str(e)}")
            logger.error("There was an issue with the OpenAI API")
            analogies.append("API Error: Please try again later")
            
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            logger.error("Attempting fallback to GPT-3.5-turbo")
            
            try:
                logger.info("Attempting to generate with GPT-3.5-turbo")
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": SYSTEM_MESSAGE},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300,
                    **generation_params[i]
                )
                analogy = response.choices[0].message.content.strip()
                logger.info("Successfully generated analogy with GPT-3.5-turbo")
                logger.debug(f"Generated analogy: {analogy[:100]}...")
                analogies.append(analogy)
                
            except Exception as fallback_error:
                logger.error(f"Fallback error: {str(fallback_error)}")
                analogies.append(f"Error generating analogy: {str(fallback_error)}")
    
    return analogies

def get_human_selection(prompt, analogies):
    """Get human selection of the best analogy"""
    print("\n=== Generated Analogies ===")
    print(f"Prompt: {prompt}\n")
    for i, analogy in enumerate(analogies, 1):
        print(f"{i}. {analogy}\n")
    
    while True:
        try:
            selection = int(input("Select the best analogy (1-3): "))
            if 1 <= selection <= len(analogies):
                # Create one-hot encoding based on selection
                one_hot = torch.zeros(len(analogies))
                one_hot[selection-1] = 1
                logger.info(f"User selected analogy {selection}")
                return one_hot, analogies[selection-1]
            print(f"Please enter a number between 1 and {len(analogies)}")
        except ValueError:
            print("Please enter a valid number")

def save_rollout(prompt, selected_analogy, one_hot_reward):
    """Save the rollout data"""
    try:
        rollout_data = {
            "prompt": prompt,
            "response": selected_analogy,
            "reward": one_hot_reward.tolist(),
            "timestamp": str(datetime.datetime.now())
        }
        
        filename = f"rollout_{torch.randint(10000, (1,)).item()}.json"
        filepath = os.path.join(ROLLOUTS_DIR, filename)
        
        with open(filepath, "w") as f:
            json.dump(rollout_data, f, indent=2)
        
        logger.info(f"Successfully saved rollout to {filename}")
        
    except Exception as e:
        logger.error(f"Error saving rollout: {str(e)}")
        print(f"Error saving rollout: {str(e)}")

def main():
    logger.info("Starting analogy generation program")
    
    while True:
        prompt = input("\nEnter a prompt (or 'quit' to exit): ")
        
        if prompt.lower() == 'quit':
            logger.info("User chose to quit")
            break
        
        logger.info(f"Processing prompt: {prompt}")
        
        # Generate multiple analogies
        analogies = generate_analogies(prompt)
        
        # Get human selection
        one_hot_reward, selected_analogy = get_human_selection(prompt, analogies)
        
        # Save the rollout
        save_rollout(prompt, selected_analogy, one_hot_reward)
        print("\n✅ Rollout saved!")

if __name__ == "__main__":
    main() 
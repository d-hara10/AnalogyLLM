import torch
import os, json
from pathlib import Path
import datetime
import logging
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

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
PROJECT_DIR = "/Users/derek/Desktop/CLUBS/AnalogyLLM/LLM Analogy"
ROLLOUTS_DIR = f"{PROJECT_DIR}/rollouts"
MODEL_DIR = f"{PROJECT_DIR}/model"
NUM_ANALOGIES = 3  # Number of analogies to generate at once

# Create directories if they don't exist
os.makedirs(ROLLOUTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# === LOAD MODEL ===
model_name = "mistralai/Mistral-7B-v0.1"

# Check if we have a valid model in the directory
has_valid_model = False
if os.path.exists(MODEL_DIR):
    try:
        # Try to load the model from the directory
        model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        has_valid_model = True
        print("Loaded existing model from", MODEL_DIR)
    except Exception as e:
        print(f"Could not load existing model: {e}")
        print("Downloading fresh model from HuggingFace...")

if not has_valid_model:
    # Load fresh model from HuggingFace
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use half precision to save memory
        device_map="auto"  # Automatically handle device placement
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        pad_token="<|pad|>",  # Set a specific pad token
        model_max_length=2048  # Set maximum length
    )
    # Save the fresh model
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print("Saved fresh model to", MODEL_DIR)

# Configure model for padding
model.config.pad_token_id = tokenizer.pad_token_id

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
            "repetition_penalty": 1.2,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id
        },
        {
            "temperature": 0.7,  # More focused
            "top_p": 0.8,
            "repetition_penalty": 1.3,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id
        },
        {
            "temperature": 1.1,  # Most creative
            "top_p": 0.95,
            "repetition_penalty": 1.1,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id
        }
    ]
    
    for i in range(num_analogies):
        try:
            logger.info(f"Attempting to generate analogy {i+1}")
            logger.debug(f"Using parameters: {generation_params[i]}")
            
            # Prepare the input with system message
            full_prompt = f"{SYSTEM_MESSAGE}\n\nUser: {prompt}\nAssistant:"
            inputs = tokenizer(
                full_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
                return_attention_mask=True
            ).to(model.device)
            
            # Generate response
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=300,
                num_return_sequences=1,
                **generation_params[i]
            )
            
            # Decode and clean the response
            analogy = tokenizer.decode(outputs[0], skip_special_tokens=True)
            analogy = analogy.replace(full_prompt, "").strip()
            
            logger.info(f"Successfully generated analogy {i+1}")
            logger.debug(f"Generated analogy: {analogy[:100]}...")  # Log first 100 chars
            analogies.append(analogy)
            
        except Exception as e:
            logger.error(f"Error generating analogy: {str(e)}")
            analogies.append(f"Error generating analogy: {str(e)}")
    
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
    logger.info("Starting analogy generation program with Mistral-7B")
    
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
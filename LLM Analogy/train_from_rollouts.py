from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig
import torch
import os, json
from pathlib import Path
import datetime

# === CONFIG ===
PROJECT_DIR = "/Users/derek/Desktop/CLUBS/AnalogyLLM/LLM Analogy"
MODEL_DIR = f"{PROJECT_DIR}/model"
ROLLOUTS_DIR = f"{PROJECT_DIR}/rollouts"

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# === LOAD MODEL ===
# Using Mistral 7B for better performance
model_name = "mistralai/Mistral-7B-v0.1"  # This is our local training model

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
        padding_side="left"
    )
    # Save the fresh model
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print("Saved fresh model to", MODEL_DIR)

# Set padding token
tokenizer.pad_token = tokenizer.eos_token

# === PPO CONFIG ===
ppo_config = PPOConfig(
    model_name=model_name,
    batch_size=2,  # Reduced batch size for memory
    forward_batch_size=1,
    log_with=None,
    learning_rate=1e-5,
    mini_batch_size=1,
)

# Initialize PPO trainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=model,
    tokenizer=tokenizer
)

def train_on_rollouts(num_epochs=3):
    # Load all rollouts
    rollout_files = list(Path(ROLLOUTS_DIR).glob("rollout_*.json"))
    if not rollout_files:
        print("No rollouts found for training!")
        return
    
    print(f"\nTraining on {len(rollout_files)} rollouts...")
    print("Note: Training on GPT-4o-mini outputs using Mistral-7B")
    print("The local model will learn from the GPT-4o-mini examples")
    
    for epoch in range(num_epochs):
        total_reward = 0
        for rollout_file in rollout_files:
            with open(rollout_file, 'r') as f:
                data = json.load(f)
            
            # Prepare inputs for PPO
            # Add system message to match GPT-4o-mini format
            system_message = """You are an expert at creating mathematical analogies across different domains. 
Your task is to explain mathematical concepts using analogies from various fields like:
- Biology (genes, cells, evolution)
- Physics (forces, energy, waves)
- Chemistry (reactions, bonds, elements)
- Computer Science (algorithms, data structures)
- Nature (ecosystems, weather, geology)
- Society (organizations, networks, growth)"""
            
            full_prompt = f"{system_message}\n\nUser: {data['prompt']}\nAssistant:"
            inputs = tokenizer(full_prompt, return_tensors="pt", padding=True).to(model.device)
            response_tokens = tokenizer(data["response"], return_tensors="pt", padding=True).to(model.device)
            
            # Get model's logits for the response
            outputs = model(**inputs, labels=response_tokens["input_ids"])
            
            # Convert one-hot reward to scalar reward
            reward_index = data["reward"].index(1) if isinstance(data["reward"], list) else 0
            reward_value = 1.0  # Positive reward for selected analogy
            
            # Create a reward tensor
            rewards = torch.tensor([reward_value], device=model.device)
            
            # Update the model using PPO
            stats = ppo_trainer.step(inputs["input_ids"], response_tokens["input_ids"], rewards)
            
            total_reward += reward_value
        
        avg_reward = total_reward / len(rollout_files)
        print(f"Epoch {epoch + 1}/{num_epochs} - Average Reward: {avg_reward:.3f}")
        
        # Save checkpoint after each epoch
        checkpoint_dir = os.path.join(MODEL_DIR, f"checkpoint-epoch-{epoch+1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"Saved checkpoint to {checkpoint_dir}")
    
    # Save the final model
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"\n✅ Final model saved to {MODEL_DIR}")
    print("Note: This is a Mistral-7B model trained on GPT-4o-mini outputs")

if __name__ == "__main__":
    train_on_rollouts() 
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig
import torch
import os, json
from pathlib import Path
import datetime

# === CONFIG ===
PROJECT_DIR = "/Users/derek/Desktop/CLUBS/LLM Analogy"
MODEL_DIR = f"{PROJECT_DIR}/model"
ROLLOUTS_DIR = f"{PROJECT_DIR}/rollouts"

# === LOAD MODEL ===
model_name = "gpt2"  # Use GPT-2 for now (small, fast)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR if os.path.exists(MODEL_DIR) else model_name)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR if os.path.exists(MODEL_DIR) else model_name)

tokenizer.pad_token = tokenizer.eos_token  # Fix padding for GPT-2

# === PPO CONFIG ===
ppo_config = PPOConfig(
    model_name=model_name,
    batch_size=4,
    forward_batch_size=2,
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
    
    for epoch in range(num_epochs):
        total_reward = 0
        for rollout_file in rollout_files:
            with open(rollout_file, 'r') as f:
                data = json.load(f)
            
            # Prepare inputs for PPO
            inputs = tokenizer(data["prompt"], return_tensors="pt", padding=True).to(model.device)
            response_tokens = tokenizer(data["response"], return_tensors="pt", padding=True).to(model.device)
            
            # Get model's logits for the response
            outputs = model(**inputs, labels=response_tokens["input_ids"])
            
            # Create a reward tensor from the saved reward
            rewards = torch.tensor([data["reward"]], device=model.device)
            
            # Update the model using PPO
            stats = ppo_trainer.step(inputs["input_ids"], response_tokens["input_ids"], rewards)
            
            total_reward += data["reward"]
        
        avg_reward = total_reward / len(rollout_files)
        print(f"Epoch {epoch + 1}/{num_epochs} - Average Reward: {avg_reward:.3f}")
    
    # Save the updated model
    model.save_pretrained(MODEL_DIR)
    print(f"\n✅ Model updated and saved to {MODEL_DIR}")

if __name__ == "__main__":
    train_on_rollouts() 
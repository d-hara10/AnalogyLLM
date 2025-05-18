# full_pipeline.py

"""
This script runs the full math analogy RLHF pipeline:
1. Prompts a language model to generate 3 analogies
2. Collects human preference as a one-hot vector
3. Converts data to pairwise format
4. Trains a reward model
5. Runs PPO to optimize the language model with TRL
"""

import torch
import random
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import Dataset
from sklearn.model_selection import train_test_split

# ========== CONFIG ==========
MODEL_NAME = "gpt2-medium"  # Better than base GPT-2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ========== 1. PROMPT THE MODEL FOR ANALOGIES ==========
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

def generate_analogies(concept, domain, num=3):
    prompt = f"Instruction: Generate 3 analogies for the math concept '{concept}' in the context of '{domain}'."
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(DEVICE)
    outputs = [
        tokenizer.decode(
            model.generate(input_ids, max_new_tokens=60, do_sample=True, top_p=0.9, top_k=50)[0],
            skip_special_tokens=True
        ) for _ in range(num)
    ]
    return outputs

# ========== 2. HUMAN PREFERENCE COLLECTION (SIMULATED HERE) ==========
def simulate_preference(analogies):
    preferred_idx = random.randint(0, 2)
    one_hot = [1 if i == preferred_idx else 0 for i in range(3)]
    return one_hot

def collect_examples(n=50):
    math_concepts = ["derivative", "entropy", "linear regression"]
    domains = ["economics", "psychology", "physics"]
    rows = []
    for _ in range(n):
        concept = random.choice(math_concepts)
        domain = random.choice(domains)
        outputs = generate_analogies(concept, domain)
        preference = simulate_preference(outputs)
        rows.append({
            "math_concept": concept,
            "target_domain": domain,
            "analogy_1": outputs[0],
            "analogy_2": outputs[1],
            "analogy_3": outputs[2],
            "preferred_1": preference[0],
            "preferred_2": preference[1],
            "preferred_3": preference[2],
        })
    return pd.DataFrame(rows)

# ========== 3. CONVERT TO PAIRWISE FORMAT ==========
def to_pairwise(df):
    pairs = []
    for _, row in df.iterrows():
        prompt = f"Generate an analogy for the math concept '{row.math_concept}' in the context of '{row.target_domain}'."
        analogies = [row.analogy_1, row.analogy_2, row.analogy_3]
        prefs = [row.preferred_1, row.preferred_2, row.preferred_3]
        chosen_idx = prefs.index(1)
        for i in range(3):
            if i != chosen_idx:
                pairs.append({"prompt": prompt, "chosen": analogies[chosen_idx], "rejected": analogies[i]})
    return pd.DataFrame(pairs)

# ========== 4. REWARD MODEL TRAINING ==========
def train_reward_model(df):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1).to(DEVICE)

    inputs = [f"{row['prompt']} Response: {row['chosen']}" for _, row in df.iterrows()] + \
             [f"{row['prompt']} Response: {row['rejected']}" for _, row in df.iterrows()]
    scores = [1.0]*len(df) + [0.0]*len(df)

    tokenized = tokenizer(inputs, truncation=True, padding=True, return_tensors='pt')
    tokenized = {k: v.to(DEVICE) for k, v in tokenized.items()}
    labels = torch.tensor(scores, dtype=torch.float).unsqueeze(1).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    model.train()
    for epoch in range(3):
        optimizer.zero_grad()
        outputs = model(**tokenized, labels=labels)
        outputs.loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1} Loss: {outputs.loss.item():.4f}")

    return model, tokenizer

# ========== 5. PPO TRAINING ==========
def run_ppo(df, reward_model, reward_tokenizer):
    config = PPOConfig(model_name=MODEL_NAME, batch_size=4, forward_batch_size=2)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_NAME).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ppo_trainer = PPOTrainer(config, model, tokenizer)

    for _, row in df.iterrows():
        prompt = row['prompt']
        query_tensor = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
        response_tensor = model.generate(query_tensor, max_new_tokens=60)[0]
        response_text = tokenizer.decode(response_tensor, skip_special_tokens=True)
        reward_input = reward_tokenizer(f"{prompt} Response: {response_text}", return_tensors="pt", truncation=True).to(DEVICE)
        reward = reward_model(**reward_input).logits[0].item()
        ppo_trainer.step([prompt], [response_text], [reward])

# ========== MAIN PIPELINE ==========
if __name__ == "__main__":
    df_examples = collect_examples(n=50)
    df_pairs = to_pairwise(df_examples)
    reward_model, reward_tokenizer = train_reward_model(df_pairs)
    run_ppo(df_pairs, reward_model, reward_tokenizer)

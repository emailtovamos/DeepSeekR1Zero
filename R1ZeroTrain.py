import re
import torch
import torch.nn.functional as F
from torch import optim
import numpy as np
import random
from transformers import LlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset
import wandb
from tqdm import tqdm
from transformers import BitsAndBytesConfig

# Initialize Weights & Biases
wandb.init(project="DeepSeek-R1-Zero-Training")

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Model configuration
model_path = "openlm-research/open_llama_3b_v2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset (small subset for testing)
dataset = load_dataset("gsm8k", "main", split="train[:20]")

# Quantization configuration for efficient memory usage
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

def preprocess_dataset(dataset):
    """Add format instructions to each example as per DeepSeek R1 Zero paper"""
    processed_data = []
    for sample in dataset:
        # Create system prompt as shown in the paper
        system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process and answer are enclosed within <think> </think> and
<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>. User: """
        
        question = sample['question']
        answer = sample['answer']
        numeric_answer = answer.split("####")[-1].strip()
        
        # Combine system prompt with question as per paper's template
        full_prompt = f"{system_prompt}{question}\nAssistant:"
        
        # Format expected response with think and answer tags
        formatted_answer = f"<think>{answer}</think>\n<answer>{numeric_answer}</answer>"
        
        processed_data.append({
            "prompt": full_prompt,
            "response": formatted_answer,
            "original_question": question,
            "expected_answer": numeric_answer
        })
    
    return processed_data

def compute_reward(response_text, expected_answer):
    """Enhanced reward function based on DeepSeek R1 Zero paper"""
    reward = 0.0
    
    # Check for presence of think tags
    has_think_start = "<think>" in response_text.lower()
    has_think_end = "</think>" in response_text.lower()
    
    # Check for presence of answer tags
    has_answer_start = "<answer>" in response_text.lower()
    has_answer_end = "</answer>" in response_text.lower()
    
    # Format reward (0.4 total for format)
    if has_think_start and has_think_end:
        reward += 0.2  # Reward for using think tags
    if has_answer_start and has_answer_end:
        reward += 0.2  # Reward for using answer tags
        
    # Extract answer and check accuracy
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.search(answer_pattern, response_text, flags=re.IGNORECASE | re.DOTALL)
    
    if match:
        extracted_answer = match.group(1).strip()
        expected_answer_cleaned = expected_answer.split("####")[-1].strip()
        
        # Accuracy reward (0.6 for correct answer)
        if extracted_answer == expected_answer_cleaned:
            reward += 0.6
    
    return reward

def evaluate_model(model, tokenizer, prompt, logging_prefix=""):
    """Evaluate model on a single prompt with detailed logging"""
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Log the response
    print(f"\n{logging_prefix}")
    print("=" * 50)
    print("Input prompt:", prompt)
    print("-" * 50)
    print("Model response:", response)
    print("=" * 50)
    
    # Extract thinking process and answer if present
    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL | re.IGNORECASE)
    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
    
    thinking = think_match.group(1).strip() if think_match else "No thinking process found"
    answer = answer_match.group(1).strip() if answer_match else "No answer found"
    
    # Log structured components
    print("\nStructured Analysis:")
    print("Thinking Process:", thinking)
    print("Final Answer:", answer)
    print("-" * 50)
    
    return response

class DeepSeekTrainer:
    def __init__(self, model, tokenizer, lr=1e-6, epsilon=0.2, group_size=4):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.epsilon = epsilon
        self.group_size = group_size
        
        print("Loading reference model...")
        self.reference_model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map='auto',
            quantization_config=quantization_config
        )
        self.reference_model.eval()
        print("Reference model loaded!")

    def train_step(self, dataset):
        self.model.train()
        total_loss = 0
        
        for batch_idx, sample in enumerate(tqdm(dataset, desc="Training...")):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            prompt = sample['prompt']
            correct_answer = sample['response']
            
            # Evaluate first sample in detail periodically
            if batch_idx == 0:
                evaluate_model(
                    self.model, 
                    self.tokenizer, 
                    prompt,
                    f"Current model response (during training batch {batch_idx})"
                )
            
            # Training logic
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            old_samples, old_rewards = [], []
            
            with torch.no_grad():
                for _ in range(self.group_size):
                    generated_ids = self.reference_model.generate(
                        input_ids,
                        max_new_tokens=200,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.95,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                    generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    reward = compute_reward(generated_text, correct_answer)
                    old_samples.append(generated_ids[:, input_ids.shape[1]:])
                    old_rewards.append(reward)

            mean_reward = np.mean(old_rewards)
            std_reward = np.std(old_rewards) if np.std(old_rewards) > 1e-6 else 1e-6
            advantages = [(r - mean_reward) / std_reward for r in old_rewards]
            
            for sample_ids, advantage in zip(old_samples, advantages):
                outputs = self.model(torch.cat([input_ids, sample_ids], dim=1))
                logits = outputs.logits[:, :-1, :]
                log_probs = F.log_softmax(logits, dim=-1)
                gathered_probs = log_probs.gather(2, sample_ids[:, :-1].unsqueeze(-1)).squeeze(-1)
                new_log_prob = gathered_probs.mean()
                
                ratio = torch.exp(new_log_prob - mean_reward)
                unclipped_objective = ratio * advantage
                clipped_objective = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantage
                policy_loss = -torch.min(unclipped_objective, clipped_objective)
                
                total_loss += policy_loss
                self.optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                wandb.log({
                    "loss": policy_loss.item(),
                    "reward": np.mean(old_rewards),
                    "advantage": advantage
                })
        
        return total_loss / len(dataset)

if __name__ == "__main__":
    print("Starting DeepSeek-R1-Zero training...")
    
    # Initialize tokenizer and model
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map='auto',
        quantization_config=quantization_config
    )
    
    # Preprocess dataset
    processed_dataset = preprocess_dataset(dataset)
    
    # Get a sample question for consistent evaluation
    test_sample = processed_dataset[0]
    print("\nSelected test sample:")
    print("Question:", test_sample['original_question'])
    print("Expected Answer:", test_sample['expected_answer'])
    
    # Evaluate base model before any training
    print("\nEvaluating base model before training:")
    evaluate_model(model, tokenizer, test_sample['prompt'], "BASE MODEL (Before Training)")
    
    # Initialize trainer
    trainer = DeepSeekTrainer(model, tokenizer, group_size=4)
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"\nStarting Epoch {epoch + 1}/{num_epochs}")
        
        # Train
        loss = trainer.train_step(processed_dataset)
        print(f"Epoch {epoch + 1} Loss: {loss:.4f}")
        
        # Evaluate after epoch
        print(f"\nEvaluating after epoch {epoch + 1}:")
        evaluate_model(model, tokenizer, test_sample['prompt'], f"MODEL AFTER EPOCH {epoch + 1}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            save_path = f"deepseek_r1_zero_epoch_{epoch + 1}"
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"Saved checkpoint to {save_path}")

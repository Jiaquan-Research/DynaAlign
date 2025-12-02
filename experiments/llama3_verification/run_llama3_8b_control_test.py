import torch
import numpy as np
import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dyna_align.controller.dyn_controller import DynControllerAgent
from dyna_align.env.goodhart_env import GoodhartEnv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_llama3():
    print("🚀 Loading Llama-3-8B-Instruct...")
    # [FIX 1] Correct 8B Model ID
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model

def llama_step(model, tokenizer, prompt, max_new_tokens=32):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_text[len(prompt):]

def run_llama3_experiment(rollout_steps=200):
    # [FIX 5] Ensure output directory exists automatically
    save_path = "experiments/data/llama3_v94_results.npz"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    tokenizer, model = load_llama3()

    # [FIX 3] V9.4 Controller Init (Removed invalid 'entropy_floor' arg)
    controller = DynControllerAgent(lr=1e-4, beta_kl=0.1)
    
    env = GoodhartEnv(seed=123)

    history_entropy = []
    history_mex = []
    history_gap = []
    
    prompt = "Explain why stability is important in AI systems:"
    print(f"🔥 Running Llama-3-8B Stability Rollout ({rollout_steps} steps)...")

    obs = env.reset()

    for t in range(rollout_steps):
        # 1) Llama generates text (Simulated Step)
        gen = llama_step(model, tokenizer, prompt)
        text_score = len(gen) / 100.0 
        
        # 2) Environment Metrics
        _, Mex, Gap = env.compute_step_metrics()
        
        # [FIX 4] CRITICAL: Sanitize NaNs from empty history at Step 0
        if np.isnan(Mex): Mex = 0.0
        if np.isnan(Gap): Gap = 0.0

        history_mex.append(Mex)
        history_gap.append(Gap)

        # 3) Controller Action (V9.4 Interface Fix)
        # [FIX 2] select_action returns (action, logp, value)
        obs_tensor = np.array([text_score, Mex, Gap], dtype=np.float32)
        
        # Pass safe tensor to network
        action, _, _ = controller.select_action(obs_tensor)
        
        # Entropy Logging (Attempt to read internal state)
        entropy_val = 0.0
        if hasattr(controller, 'entropy_history') and len(controller.entropy_history) > 0:
            entropy_val = controller.entropy_history[-1]
            
        history_entropy.append(entropy_val)

        # 4) Step Env
        obs, _ = env.step(action)

        if t % 10 == 0:
            print(f"Step {t:4d} | Mex={Mex:.3f} | Gap={Gap:.3f} | Ent={entropy_val:.3f}")

    print("📁 Saving logs:", save_path)
    np.savez(save_path, entropy=history_entropy, mex=history_mex, gap=history_gap)
    print("✅ Done.")

if __name__ == "__main__":
    run_llama3_experiment()
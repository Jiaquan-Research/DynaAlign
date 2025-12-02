# Llama-3-8B Real Run (Run #1)
# Environment Setup and Reproduction Log
Date: 2025-11-30
Author: He Jiaqian

This document describes the exact steps used in the **first successful real 8B run**
executed on RunPod using Meta Llama-3-8B-Instruct.

The goal of this document is:
- Reproducibility
- Engineering traceability
- Allow future reviewers to replicate the environment without ambiguity

---

## 1. Pod Spec (RunPod)

GPU: A6000
Storage: 100GB persistent volume  
Framework: RunPod Base Image (NVIDIA CUDA 12.x)

---

## 2. Login and Authentication

SSH login command (Windows PowerShell):

```

ssh <pod-id>@ssh.runpod.io -i C:\Users\Administrator.ssh\id_ed25519

```

HuggingFace Login:

```

huggingface-cli login

````

Confirm token:

```python
from huggingface_hub import login
print("HF Token OK")
````

---

## 3. Install Dependencies

```
pip install hf_transfer
pip install matplotlib==3.8.2
pip install transformers==4.45.0
pip install accelerate
```

---

## 4. Download Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
print("Model Loaded!")
```

---

## 5. Run Script (Original version)

```
python experiments/llama3_verification/run_llama3_8b_control_test.py
```

---

## 6. Results (First Run)

* Model successfully loaded
* Controller intervention appeared correct
* Output text saved
* No crash during inference

Artifacts saved under:

```
runs/llama3_control_real/<timestamp>
```

---

## 7. Notes from Run #1

* This was the first successful Llama-3-8B run on GPU
* Demonstrated viability of Phoenix for inference-time governance
* Provided baseline for later “research-grade” script




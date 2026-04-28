from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Path to your unzipped fine-tuned model
model_path = r"D:\\VPS_WEB\\deepseek_patient_lora"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load model (use device_map for GPU if available)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

# Inference function
def ask_patient(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example doctor-patient simulation
doctor_prompt = """
Doctor: Good morning. What brings you in today?
Patient:
"""

print(ask_patient(doctor_prompt))

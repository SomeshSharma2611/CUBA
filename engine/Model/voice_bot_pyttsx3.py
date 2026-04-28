import torch
import speech_recognition as sr
import pyttsx3
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Paths and model names
clinical_bert_path = "./clinicalbert_finetuned"  # Update to your fine-tuned ClinicalBERT path
phi3_mini_model_name = "microsoft/Phi-3-mini-4k-instruct"
cache_dir = "D:\\VPS\\phi3_cache"

# Load ClinicalBERT for classification
print("Loading ClinicalBERT...")
clinical_bert_tokenizer = AutoTokenizer.from_pretrained(clinical_bert_path)
clinical_bert_model = AutoModelForSequenceClassification.from_pretrained(clinical_bert_path).to(device)

# Load Phi-3-mini for chatbot responses
print("Loading Phi-3-mini...")
phi3_mini_tokenizer = AutoTokenizer.from_pretrained(phi3_mini_model_name, cache_dir=cache_dir)
phi3_mini_model = AutoModelForCausalLM.from_pretrained(
    phi3_mini_model_name,
    cache_dir=cache_dir,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else "auto"
)

# Compile the model (optional, for PyTorch 2.0+)
try:
    phi3_mini_model = torch.compile(phi3_mini_model)
except:
    print("torch.compile() is not supported, skipping optimization.")

# Text-to-Speech using pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 165)  # Optional: Set speech rate

def say(text):
    engine.say(text)
    engine.runAndWait()

# Speech-to-Text using SpeechRecognition
def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 0.6
        r.adjust_for_ambient_noise(source)
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            print("Listening timed out while waiting for phrase to start")
            return ""  # No input detected at all

        try:
            print("Recognizing...")
            query = r.recognize_google(audio, language="en-in")
            print(f"Doctor said: {query}")
            return query
        except sr.UnknownValueError:
            print("Speech Recognition could not understand audio")
            return ""
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return ""


# Function to classify input using ClinicalBERT
def classify_input(text):
    inputs = clinical_bert_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128).to(device)
    outputs = clinical_bert_model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

# Function to generate chatbot response using Phi-3-mini
def generate_response(conversation_history):
    input_text = phi3_mini_tokenizer.apply_chat_template(conversation_history, tokenize=False, add_generation_prompt=True)
    inputs = phi3_mini_tokenizer(input_text, return_tensors="pt").to(phi3_mini_model.device)

    with torch.no_grad():
        outputs = phi3_mini_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=75,
            do_sample=True,
            top_k=40,
            top_p=0.85,
            temperature=0.6
        )

    response = phi3_mini_tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()

# Initialize chat history
conversation_history = [
    {
        "role": "system",
        "content": "You are a virtual patient with a realistic Indian profile. Always respond as if you are experiencing the symptoms firsthand. Keep your responses brief and realistic, addressing only what the doctor asks. Do not over-explain or give unnecessary information. Never mention that you are an AI or a virtual patient. Stay in character and react naturally, as a real patient would."
    }
]

# Chatbot loop
print("Patient Chatbot: Hello doctor!")
say("Hello doctor!")
conversation_started = False  # Track if the conversation has begun

while True:
    # Take voice input from the doctor
    user_input = takeCommand()

    if user_input.lower() in ["exit", "quit", "stop"]:
        print("Patient Chatbot: Goodbye doctor!")
        say("Goodbye doctor!")
        break

    if not user_input:
        # Handle first-time silence differently
        if not conversation_started:
            print("Patient Chatbot: Doctor, I'm waiting for your question.")
            say("Doctor, I'm waiting for your question.")
        else:
            print("Patient Chatbot: I couldn't hear you properly, could you please repeat?")
            say("I couldn't hear you properly, could you please repeat?")
        continue

    conversation_started = True # Set flag once valid input is received

    # Classify the doctor's input using ClinicalBERT,  Classify only when valid input is present
    classification = classify_input(user_input)

    if classification == 0:
        # If the input is irrelevant, respond accordingly
        response = "Doctor, that question doesn't seem relevant to my medical condition."
    else:
        # If the input is relevant, generate a response
        conversation_history.append({"role": "user", "content": user_input})
        response = generate_response(conversation_history)
        conversation_history.append({"role": "assistant", "content": response})

    # Print and say the patient response
    print(f"Patient Chatbot: {response}")
    say(response)

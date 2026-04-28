import eel
import torch
import speech_recognition as sr
import pyttsx3
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
import time

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Paths and model names
clinical_bert_path = "./engine/Model/clinicalbert_finetuned"  # Update to your fine-tuned ClinicalBERT path
phi3_mini_model_name = "microsoft/Phi-3-mini-4k-instruct"
cache_dir = "D:\\VPS_WEB\\engine\\Model\\phi3_cache"

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


  

def say(text):
    engine = pyttsx3.init()# Text-to-Speech using pyttsx3
    engine.setProperty('rate', 165)# Optional: Set speech rate
    engine.say(text)
    engine.runAndWait()
# def say(text):
#     eel.showPatientTalking()  # Switch to animated GIF
#     engine = pyttsx3.init()
#     engine.setProperty('rate', 165)
#     engine.say(text)
#     engine.runAndWait()
#     eel.showPatientListening()  # Switch back to static image


# Speech-to-Text using SpeechRecognition
def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        eel.DisplayMessage("Listening...")
        r.pause_threshold = 0.6
        r.adjust_for_ambient_noise(source)
        try:
            audio = r.listen(source, timeout=6, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            print("Listening timed out while waiting for phrase to start")
            return ""  # No input detected at all

        try:
            print("Recognizing...")
            eel.DisplayMessage("Recognizing")
            query = r.recognize_google(audio, language="en-in")
            print(f"Doctor said: {query}")
            eel.DisplayMessage(query)
            time.sleep(2)
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
        "content": "You are an Indian patient currently experiencing real health symptoms. Respond naturally and concisely, just as a real patient would during a medical consultation. Answer only what the doctor asks—do not offer additional details or explanations unless prompted. Your responses should reflect the tone, clarity, and emotions of someone genuinely experiencing the symptoms. Stay fully in character and never reveal or imply that you are not a real person."
    }
]

# Chatbot loop
# print("Patient Chatbot: Hello doctor!")
# say("Hello doctor!")
conversation_started=False

@eel.expose
def startSession():
    global conversation_started
    """
    Runs a loop: listen → classify → generate → speak → send back to JS
    Stops when the user says 'exit', 'quit', or 'stop', then returns control.
    """
    
    #eel.DisplayMessage("Hello Doctor")
    print("Patient Chatbot: Hello Doctor")
    say("Hello Doctor")
    

    while True:
        # 1. Listen for doctor
        user_input = takeCommand()

        # 2. If exit phrase, send farewell and break
        if user_input.lower() in ("exit", "quit", "stop"):
            farewell = "Goodbye doctor!"
            print("Patient Chatbot:"+farewell)
            eel.DisplayMessage(farewell)
            say(farewell)
            return  # Return to JS, ending the session

        # 3. No input or misheard
        if not user_input:
            if not conversation_started:
                prompt = "Doctor, I'm waiting for your question."
            else:
                prompt = "I couldn't listen it, Could you please repeat that?"
            print(prompt)
            eel.DisplayMessage(prompt)
            say(prompt)
            continue

        #once we get valid input:
        conversation_started=True

        # 4. Classify relevance
        if classify_input(user_input) == 0:
            response = "Doctor, that question doesn't seem relevant to my condition."
        else:
            conversation_history.append({"role": "user", "content": user_input})
            response = generate_response(conversation_history)
            conversation_history.append({"role": "assistant", "content": response})

        # 5. Display & speak the response
        print(f"Patient Chatbot: {response}")
        eel.DisplayMessage(response)
        say(response)
        # then loop back for next doctor question
0
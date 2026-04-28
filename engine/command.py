import eel
from engine.voice_utils import listen, speak
from engine.classifier_bert import classify
from engine.llm_phi3 import generate_response

# Conversation state
conversation_history = [{
    "role": "system",
    "content": (
        "You are a virtual patient with a realistic Indian profile. Always respond as if you are experiencing the symptoms firsthand. Keep your responses brief and realistic, addressing only what the doctor asks. Do not over-explain or give unnecessary information. Never mention that you are an AI or a virtual patient. Stay in character and react naturally, as a real patient would."
    )
}]

conversation_started=False

@eel.expose
def startSession():
    global conversation_started
    """
    Runs a loop: listen → classify → generate → speak → send back to JS
    Stops when the user says 'exit', 'quit', or 'stop', then returns control.
    """
    while True:
        # 1. Listen for doctor
        user_input = listen()

        # 2. If exit phrase, send farewell and break
        if user_input.lower() in ("exit", "quit", "stop"):
            farewell = "Goodbye doctor!"
            print("Patient Chatbot:"+farewell)
            eel.DisplayMessage(farewell)
            speak(farewell)
            return  # Return to JS, ending the session

        # 3. No input or misheard
        if not user_input:
            if not conversation_started:
                prompt = "Doctor, I'm waiting for your question."
            else:
                prompt = "Could you please repeat that?"
            print(prompt)
            eel.DisplayMessage(prompt)
            speak(prompt)
            continue

        #once we get valid input:
        conversation_started=True

        # 4. Classify relevance
        if classify(user_input) == 0:
            response = "Doctor, that question doesn't seem relevant to my condition."
        else:
            conversation_history.append({"role": "user", "content": user_input})
            response = generate_response(conversation_history)
            conversation_history.append({"role": "assistant", "content": response})

        # 5. Display & speak the response
        print(f"Patient Chatbot: {response}")
        eel.DisplayMessage(response)
        speak(response)
        # then loop back for next doctor question

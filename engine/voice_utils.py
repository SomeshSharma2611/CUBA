import speech_recognition as sr
import pyttsx3
import eel
import time

def listen(timeout=5, phrase_time_limit=10):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        eel.DisplayMessage("Listening...")
        r.pause_threshold = 0.6
        r.adjust_for_ambient_noise(source)

        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
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

def speak(text, rate=165):
    engine = pyttsx3.init()
    engine.setProperty('rate', rate)
    engine.say(text)
    engine.runAndWait()

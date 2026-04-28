import os
import eel
from engine.features import playBotSound
from engine.voice_bot_pyttsx3_full_version import *
eel.init("www")
playBotSound()
os.system('start msedge.exe --app="http://localhost:8000/index.html"')
eel.start('index.html',mode=None,host="localhost",block=True)
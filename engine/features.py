from playsound import playsound
import eel


@eel.expose
def playBotSound():
    music_dir="D:\\VPS_WEB\\www\\assets\\Audio\\start_sound.mp3"
    playsound(music_dir)


@eel.expose
def MicSound():
    music_dir="D:\\VPS_WEB\\www\\assets\\Audio\\mic_button_sound.wav"
    playsound(music_dir)
import os
import torch
import soundfile as sf
import google.generativeai as genai
import requests
import numpy as np
import sounddevice as sd
import tempfile
import re
from pynput import keyboard
from transformers import pipeline
from dotenv import load_dotenv

# --- KONFIGURASJON ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Mangler GEMINI_API_KEY i .env-filen.")
genai.configure(api_key=GEMINI_API_KEY)

chat_session = genai.GenerativeModel("gemini-2.5-flash").start_chat(history=[])
print("✅ Samisk stemmeassistent er klar.")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipe_speech_to_sami_text = pipeline(
    "automatic-speech-recognition",
    model="NbAiLab/whisper-large-sme",
    device=device,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
print(f"✅ Whisper-modell lastet inn på {device}.")

SAMPLE_RATE = 16000
CHANNELS = 1
PTT_KEY = keyboard.Key.space

# --- PTT-VARIABLER OG FUNKSJONER ---
is_recording = False
recorded_frames = []

def on_press(key):
    global is_recording, recorded_frames
    if key == PTT_KEY and not is_recording:
        print("\n" + "="*40)
        print("🎤 Lytting startet...")
        is_recording = True
        recorded_frames = []

def on_release(key):
    global is_recording
    if key == PTT_KEY and is_recording:
        print("🛑 Lytting stoppet. Prosesserer...")
        is_recording = False
        return False
    return True

def record_audio_with_ptt():
    """PTT-opptaksfunksjon."""
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    print("Hold inne [MELLOMROM] for å snakke...")
    
    def audio_callback(indata, frames, time, status):
        if is_recording:
            recorded_frames.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback):
        listener.join()

    if not recorded_frames:
        print("Ingen lyd ble tatt opp.")
        return None

    recording = np.concatenate(recorded_frames, axis=0)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    sf.write(temp_file.name, recording, SAMPLE_RATE)
    return temp_file.name

# --- KJERNEFUNKSJONER ---

def is_valid_charset(text: str) -> bool:
    """Sjekker om teksten kun inneholder gyldige norsk/samiske tegn."""
    # Tillater a-z, norske tegn, samiske tegn, tall, og vanlig tegnsetting.
    # Finner alle tegn som IKKE er i dette settet.
    invalid_char_pattern = r"[^a-zA-ZæøåÆØÅáčđŋšŧžÁČĐŊŠŦŽ0-9\s.,;:!?\"'()\[\]-]"
    match = re.search(invalid_char_pattern, text)
    # Returnerer True hvis ingen ugyldige tegn ble funnet.
    return match is None

def clean_markdown_text(text: str) -> str:
    """Renser tekst for Markdown og gjør den klar for oversettelse og TTS."""
    print("🧼 Vasker tekst...")
    text = re.sub(r'[\*_]{1,2}(.+?)[\*_]{1,2}', r'\1', text)
    text = re.sub(r'^\s*[\*\-]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\#+\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    text = re.sub(r'`{1,3}(.*?)`{1,3}', r'\1', text)
    text = re.sub(r'\n\s*\n', '. ', text)
    text = text.replace('\n', ' ')
    return text.strip()

def generate_sami_speech(text: str):
    """Bruker Giellatekno sin web-API for å lage tale."""
    print("🔊 Genererer samisk tale...")
    api_url = "https://api-giellalt.uit.no/tts/se/biret"
    payload = {'text': text}
    try:
        response = requests.post(api_url, json=payload, timeout=20)
        response.raise_for_status()
        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_audio_file.write(response.content)
        temp_audio_file.close()
        return temp_audio_file.name
    except requests.exceptions.RequestException as e:
        print(f"❌ Feil med talesyntese-API: {e}")
        return None

def play_audio(filename: str):
    """Bruker soundfile og sounddevice for å spille av lyd."""
    try:
        data, samplerate = sf.read(filename, dtype='float32')
        print("▶️ Spiller av svar...")
        sd.play(data, samplerate)
        sd.wait()
    except Exception as e:
        print(f"Kunne ikke spille av lydfil: {e}")
    finally:
        if os.path.exists(filename):
            os.remove(filename)

def translate_text(text: str, source_lang: str, target_lang: str) -> str | None:
    """Oversetter tekst med TartuNLP API, med retry-logikk."""
    api_url = "https://api.tartunlp.ai/translation/v2"
    payload = {"text": text, "src": source_lang, "tgt": target_lang}
    headers = {"Content-Type": "application/json"}
    
    # Løkke for å prøve opptil 2 ganger
    for attempt in range(2):
        print(f"🔄 Oversetter fra {source_lang} til {target_lang} (forsøk {attempt + 1})...")
        try:
            response = requests.post(api_url, json=payload, headers=headers, timeout=15)
            response.raise_for_status()
            translated_text = response.json().get("result")

            if not translated_text or not isinstance(translated_text, str) or not translated_text.strip():
                print("⚠️ Oversettelsen ga et tomt resultat.")
                continue # Gå til neste forsøk

            if not is_valid_charset(translated_text):
                print("⚠️ Oversettelsen inneholdt ugyldige tegn. Prøver på nytt...")
                continue # Gå til neste forsøk

	    # Feiler på 'Ok.' og lignende
            #if translated_text.strip() == text.strip():
            #    print("⚠️ Oversettelsen returnerte input-teksten uendret.")
            #    continue # Gå til neste forsøk
            
            # Hvis alt er OK, returner resultatet
            return translated_text

        except requests.exceptions.RequestException as e:
            print(f"❌ Feil med oversettelses-API: {e}")
    
    # Hvis løkken fullføres uten et gyldig resultat
    print(f"❌ Klarte ikke å få en gyldig oversettelse for '{text}' etter 2 forsøk.")
    return None

def process_sami_audio(audio_file_path: str):
    if not audio_file_path: return

    # STEG 1 & 2: Tale -> Samisk tekst -> Norsk tekst
    print("🗣️  Transkriberer...")
    result = pipe_speech_to_sami_text(audio_file_path, generate_kwargs={"task": "transcribe"})
    os.remove(audio_file_path)
    sami_text_from_speech = result["text"]
    if not sami_text_from_speech or not sami_text_from_speech.strip():
        print("Ingen gjenkjennelig tale."); return
    print(f" Sámi: '{sami_text_from_speech}'")
    norwegian_text = translate_text(sami_text_from_speech, "sme", "nor")
    if not norwegian_text:
        print("Kunne ikke oversette til norsk."); return
    print(f"🇳🇴  Oversatt: '{norwegian_text}'")

    # STEG 3: Norsk tekst -> Gemini-svar
    print("🧠 Tenker...")
    try:
        prompt = f"Svar kort og på et enkelt, muntlig norsk. Svaret skal leses opp av en stemmeassistent. Spørsmål: {norwegian_text}"
        response = chat_session.send_message(prompt)
        gemini_response_norwegian = response.text
    except Exception as e:
        print(f"En feil oppstod med Gemini: {e}"); return

    # STEG 4 & 5: Vask, oversett tilbake og generer tale
    cleaned_norwegian_text = clean_markdown_text(gemini_response_norwegian)
    final_sami_text = translate_text(cleaned_norwegian_text, "nor", "sme")

    if final_sami_text:
        audio_file = generate_sami_speech(final_sami_text)
        if audio_file:
            print(f" Sámi (svar): {final_sami_text}")
            play_audio(audio_file)
        else:
            print("\n⚠️ Kunne ikke generere tale. Viser tekst-svar:")
            print(f" Sámi: {final_sami_text}")
    else:
        print("\n⚠️ Kunne ikke oversette til samisk. Viser norsk svar:")
        print(f"🇳🇴: {cleaned_norwegian_text}")

# --- HOVED-LOOP ---
if __name__ == "__main__":
    try:
        while True:
            audio_file = record_audio_with_ptt()
            if audio_file:
                process_sami_audio(audio_file)
    except KeyboardInterrupt:
        print("\nAvslutter programmet.")
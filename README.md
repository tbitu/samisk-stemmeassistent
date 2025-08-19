# Samisk stemmeassistent

Dette er en Python-basert stemmeassistent som lytter til nordsamisk tale, oversetter det til norsk for å få et svar fra Gemini, og oversetter svaret tilbake til samisk tale.

## Funksjoner
* **Tale-til-Tekst:** Bruker Nasjonalbibliotekets Whisper-modell for å transkribere samisk tale.
* **Oversettelse:** Bruker TartuNLP for oversettelse mellom samisk og norsk.
* **AI-motor:** Bruker Google Gemini for å generere svar.
* **Tekst-til-Tale:** Bruker Giellatekno (UiT) sin API for å generere samisk tale.
* **Push-to-Talk:** Bruker mellomromstasten for å aktivere mikrofonen.

## Installasjon og bruk
1.  Installer alle nødvendige pakker:
    *For GPU-støtte, installer PyTorch først:*
    ```
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
    pip install -r requirements.txt
    ```

2.  Lag en `.env`-fil i samme mappe og legg til din Gemini API-nøkkel:
    ```
    GEMINI_API_KEY=DIN_API_NØKKEL_HER
    ```

3.  Kjør skriptet:
    ```
    python main.py
    ```

4.  Hold inne mellomromstasten for å snakke.


Hvis du har en Nvidia GPU, kan du sjekke om scriptet kan bruke det med cudatest.py.

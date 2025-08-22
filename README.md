# Samisk stemmeassistent

Dette er en Python-basert stemmeassistent som lytter til nordsamisk og norsk tale, transkriberer det automatisk til norsk tekst, benytter AI for å generere svar, oversetter svaret til samisk, og leser det opp.

## Funksjoner
* **Tale-til-Tekst:** Bruker nb-whisper-large-modellen (Nasjonalbiblioteket) til å transkribere både samisk og norsk tale direkte til norsk tekst.
* **AI-motor:** Bruker Google Gemini for å generere svar på norsk.
* **Oversettelse:** Bruker TartuNLP for å oversette Gemini sitt norske svar til samisk.
* **Tekst-til-Tale:** Bruker Giellatekno (UiT) sin API for å generere samisk tale.

## Installasjon og bruk
1.  Installer alle nødvendige pakker:
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

4.  Trykk Enter for å starte og stoppe lydopptak.

Hvis du har en Nvidia GPU, kan du sjekke om scriptet kan bruke det med cudatest.py.

# AI Chatbot

An AI chatbot built using Python, NLP and TensorFlow.

## Features
- Intent recognition
- NLP preprocessing
- Response generation
- AI image generation from a text prompt
- Device image upload and AI prediction

## Technologies Used
- Python
- TensorFlow
- NLTK
- NumPy
- Google Gemini API

## Setup
1. Install the Python packages:
   `pip install -r requirements.txt`
2. Copy `.env.example` to `.env`.
3. Put your Gemini API key in `.env`.
4. Start the browser chatbot:
   `python DATA/main.py`

The command starts a local web app at `http://127.0.0.1:8000` and opens it in your default browser.
Use `python DATA/main.py --cli` for the terminal chatbot or `python DATA/UI.py` for the Tkinter desktop app.

## Image Commands
- `/image a futuristic city at sunset` generates and previews an image.
- `/analyze` opens a device picker and predicts what the selected image contains.
- `/analyze Is there a vehicle in this image?` asks a custom question about a selected image.

The normal NLP chatbot still works without an API key. Gemini image tools load only when you use an image command.

## Author
Dhruva Parode
SIH 2025 Finalist

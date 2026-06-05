# 🤖 NLP Chatbot — Internship Project

A full-featured AI chatbot built with **Python + TensorFlow + Google Gemini**.  
Supports three interfaces: **Web App**, **Tkinter Desktop UI**, and **Terminal CLI**.

---

## 📁 Project Structure

```
chatbot/
├── main.py               # Core chatbot engine (NLP model + commands)
├── web_app.py            # Built-in HTTP web server
├── UI.py                 # Tkinter desktop GUI
├── ai_image_service.py   # Google Gemini image generation & analysis
├── intents.json          # Training data (add more Q&A pairs here!)
├── chatbot_model.h5      # Trained TensorFlow model
├── classes.pkl           # Intent class labels
├── words.pkl             # Vocabulary (auto-generated on train)
├── requirements.txt      # Python dependencies
├── .env                  # Your secret API key (never share this!)
├── web/
│   └── index.html        # Web app frontend
└── generated_images/     # AI-generated images saved here
```

---

## ⚙️ Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure your API key
Open `.env` and replace the placeholder:
```
GOOGLE_API_KEY=your_real_key_here
```
Get a free key at: https://aistudio.google.com/app/apikey

### 3. Retrain the model (if you edited intents.json)
```bash
python main.py --train
```

---

## 🚀 Running the Chatbot

### Web App (recommended)
```bash
python main.py
```
Opens at http://127.0.0.1:8000 automatically.

### Desktop GUI
```bash
python UI.py
```

### Terminal / CLI
```bash
python main.py --cli
```

---

## 💬 Available Commands

| Command              | Description                        |
|----------------------|------------------------------------|
| `/help`              | Show all commands                  |
| `/time`              | Show current time                  |
| `/date`              | Show today's date                  |
| `/search <query>`    | Open a Google search               |
| `/image <prompt>`    | Generate an image with Gemini AI   |
| `/analyze`           | Upload & analyze an image          |
| `/clear`             | Clear the chat history (web/UI)    |
| `/save`              | Save chat to a .txt file (UI)      |
| `bye`                | End the conversation               |

---

## ✏️ Adding New Intents

Edit `intents.json` — add a new block like this:

```json
{
  "tag": "about_internship",
  "patterns": [
    "What is this project",
    "Tell me about the chatbot",
    "What did you build"
  ],
  "responses": [
    "This is an NLP chatbot built during my internship using TensorFlow and Python!",
    "It's an AI chatbot that understands natural language using a trained neural network."
  ]
}
```

Then retrain:
```bash
python main.py --train
```

---

## 🛠 Tech Stack

- **Python 3.10+**
- **TensorFlow / Keras** — neural network for intent classification
- **NLTK** — tokenization and lemmatization
- **Google Gemini API** — image generation & vision analysis
- **Tkinter** — desktop UI
- **Built-in HTTP server** — no Flask needed for the web app

---

## 🔧 Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `No Google API key` | Set `GOOGLE_API_KEY` in `.env` |
| Model gives wrong answers | Add more patterns to `intents.json` and retrain |
| `words.pkl` not found | Run `python main.py --train` |

---

*Built for internship use — feel free to extend intents, style the UI, or deploy to a cloud server!*

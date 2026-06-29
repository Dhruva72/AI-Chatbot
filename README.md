# NLP Chatbot with local Ollama

A Python, TensorFlow, and Ollama chatbot with web, Tkinter, and terminal interfaces. General questions are answered locally by `llama3`; no cloud API key is required.

## Setup

1. Install the Python dependencies:

   ```powershell
   .\.conda\python.exe -m pip install -r requirements.txt
   ```

2. Start Ollama and install the text model:

   ```powershell
   ollama serve
   ollama pull llama3
   ```

3. Optionally copy `.env.example` to `.env` and change the model or endpoint:

   ```dotenv
   OLLAMA_URL=http://127.0.0.1:11434/api/generate
   OLLAMA_MODEL=llama3
   ```

4. Run the web chatbot from the project root:

   ```powershell
   .\.conda\python.exe DATA\main.py
   ```

   It opens at [http://127.0.0.1:8000](http://127.0.0.1:8000).

   You can also double-click `run_chatbot.bat`; it always uses the project
   `.conda` Python instead of Anaconda `base` or a global Python install.

For terminal mode, run `.\.conda\python.exe DATA\main.py --cli`. For the desktop UI, run `.\.conda\python.exe DATA\UI.py`.

## Commands

| Command | Description |
|---|---|
| `/help` | Show commands |
| `/ask <question>` | Ask the local Llama 3 model directly |
| `/mood` | Show detected sentiment |
| `/time`, `/date` | Show local time or date |
| `/search <query>` | Open an explicitly requested web search |
| `/analyze` | Analyze an upload with an optional Ollama vision model |
| `/image <prompt>` | Explain the local text model's image-generation limitation |

## Optional image analysis

The installed `llama3` model is text-only. To analyze uploaded images, install an Ollama vision model and configure it:

```powershell
ollama pull llama3.2-vision
```

```dotenv
OLLAMA_VISION_MODEL=llama3.2-vision
```

Ollama language and vision models do not generate PNG files from text, so `/image` intentionally returns a clear explanation instead of calling a cloud service.

## Troubleshooting

| Problem | Fix |
|---|---|
| Ollama is not reachable | Start the Ollama application/service and confirm port `11434` is available |
| Model not found | Run `ollama pull llama3`, or set `OLLAMA_MODEL` to an installed model |
| Vision model not found | Run `ollama pull llama3.2-vision`, or set `OLLAMA_VISION_MODEL` |
| Python module missing | Use `.\.conda\python.exe DATA\main.py`, or run `.\.conda\python.exe -m pip install -r requirements.txt` |
| Intent model is stale | Run `.\.conda\python.exe DATA\main.py --train` |

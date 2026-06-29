import datetime as dt
import threading
import webbrowser
from pathlib import Path

try:
    from project_env import ensure_project_python
except ImportError:
    from .project_env import ensure_project_python

ensure_project_python()

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

try:
    from .ai_image_service import AIImageService
    from .main import ChatbotEngine
except ImportError:
    from ai_image_service import AIImageService
    from main import ChatbotEngine


APP_DIR = Path(__file__).resolve().parent


class ChatbotProfileUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.user_name = tk.StringVar(value="User")
        self.status_text = tk.StringVar(value="Loading chatbot...")
        self.conversation_history: list[tuple[str, str]] = []
        self.ai_service: AIImageService | None = None
        self.ai_busy = False
        self.ai_buttons: list[ttk.Button] = []
        self.generated_image_refs: list[tk.PhotoImage] = []

        self.configure_window()
        self.build_layout()
        self.load_engine()

    def configure_window(self) -> None:
        self.root.title("NLP Chatbot - User Console")
        self.root.geometry("980x680")
        self.root.minsize(820, 560)
        self.root.configure(bg="#f6f7fb")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#f6f7fb")
        style.configure("Panel.TFrame", background="#ffffff", relief="flat")
        style.configure("TLabel", background="#f6f7fb", foreground="#202534", font=("Segoe UI", 10))
        style.configure("Panel.TLabel", background="#ffffff", foreground="#202534", font=("Segoe UI", 10))
        style.configure("Title.TLabel", background="#ffffff", foreground="#111827", font=("Segoe UI", 16, "bold"))
        style.configure("Subtle.TLabel", background="#ffffff", foreground="#667085", font=("Segoe UI", 9))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))

    def build_layout(self) -> None:
        shell = ttk.Frame(self.root, padding=16)
        shell.pack(fill=tk.BOTH, expand=True)
        shell.columnconfigure(1, weight=1)
        shell.rowconfigure(0, weight=1)

        sidebar = ttk.Frame(shell, style="Panel.TFrame", padding=16)
        sidebar.grid(row=0, column=0, sticky="ns", padx=(0, 14))
        sidebar.columnconfigure(0, weight=1)

        ttk.Label(sidebar, text="User Profile", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(sidebar, text="Your personal chatbot command center", style="Subtle.TLabel").grid(
            row=1, column=0, sticky="w", pady=(2, 16)
        )

        ttk.Label(sidebar, text="Display name", style="Panel.TLabel").grid(row=2, column=0, sticky="w")
        name_row = ttk.Frame(sidebar, style="Panel.TFrame")
        name_row.grid(row=3, column=0, sticky="ew", pady=(6, 16))
        name_row.columnconfigure(0, weight=1)

        self.name_entry = ttk.Entry(name_row, textvariable=self.user_name, width=24)
        self.name_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(name_row, text="Apply", command=self.apply_profile_name).grid(row=0, column=1)

        ttk.Label(sidebar, text="Commands", style="Panel.TLabel").grid(row=4, column=0, sticky="w")
        self.command_frame = ttk.Frame(sidebar, style="Panel.TFrame")
        self.command_frame.grid(row=5, column=0, sticky="nsew", pady=(8, 16))

        ttk.Label(sidebar, textvariable=self.status_text, style="Subtle.TLabel", wraplength=250).grid(
            row=6, column=0, sticky="ew", pady=(8, 0)
        )
        sidebar.rowconfigure(5, weight=1)

        chat_panel = ttk.Frame(shell, style="Panel.TFrame", padding=16)
        chat_panel.grid(row=0, column=1, sticky="nsew")
        chat_panel.columnconfigure(0, weight=1)
        chat_panel.rowconfigure(1, weight=1)

        header = ttk.Frame(chat_panel, style="Panel.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Ask the Chatbot", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Type a question, run a command, or use the shortcuts from your profile.",
            style="Subtle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        self.chat_history = scrolledtext.ScrolledText(
            chat_panel,
            wrap=tk.WORD,
            font=("Segoe UI", 10),
            bg="#fbfcfe",
            fg="#202534",
            relief=tk.FLAT,
            padx=12,
            pady=12,
        )
        self.chat_history.grid(row=1, column=0, sticky="nsew")
        self.chat_history.tag_configure("user", foreground="#175cd3", font=("Segoe UI", 10, "bold"))
        self.chat_history.tag_configure("bot", foreground="#087443", font=("Segoe UI", 10, "bold"))
        self.chat_history.tag_configure("meta", foreground="#667085", font=("Segoe UI", 9))
        self.chat_history.configure(state=tk.DISABLED)

        input_area = ttk.Frame(chat_panel, style="Panel.TFrame")
        input_area.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        input_area.columnconfigure(0, weight=1)

        self.user_input = ttk.Entry(input_area, font=("Segoe UI", 11))
        self.user_input.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.user_input.bind("<Return>", lambda _event: self.send_message())

        ttk.Button(input_area, text="Send", style="Accent.TButton", command=self.send_message).grid(row=0, column=1)

        actions = ttk.Frame(chat_panel, style="Panel.TFrame")
        actions.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        generate_button = ttk.Button(actions, text="Generate Image", command=self.generate_image_from_input)
        generate_button.pack(side=tk.LEFT, padx=(0, 8))
        analyze_button = ttk.Button(actions, text="Upload & Predict", command=lambda: self.send_message("/analyze"))
        analyze_button.pack(side=tk.LEFT, padx=(0, 8))
        self.ai_buttons.extend([generate_button, analyze_button])
        ttk.Button(actions, text="Clear Chat", command=self.clear_chat).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(actions, text="Save Chat", command=self.save_chat).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(actions, text="Help", command=lambda: self.send_message("/help")).pack(side=tk.LEFT)

    def load_engine(self) -> None:
        try:
            self.engine = ChatbotEngine()
        except Exception as exc:
            self.status_text.set("Chatbot could not load. Check model and dependency files.")
            messagebox.showerror("Chatbot Load Error", str(exc))
            self.engine = None
            return

        self.status_text.set("Ready. Choose a command or ask a question.")
        self.render_command_buttons()
        self.add_bot_message("Hello! I am ready to help. Type /help to see everything I can do.")
        self.user_input.focus_set()

    def render_command_buttons(self) -> None:
        for child in self.command_frame.winfo_children():
            child.destroy()

        for row, (command, description) in enumerate(self.engine.commands):
            button = ttk.Button(
                self.command_frame,
                text=command,
                command=lambda selected=command: self.use_command(selected),
            )
            button.grid(row=row, column=0, sticky="ew", pady=3)
            ttk.Label(
                self.command_frame,
                text=description,
                style="Subtle.TLabel",
                wraplength=240,
            ).grid(row=row, column=1, sticky="w", padx=(8, 0), pady=3)
        self.command_frame.columnconfigure(1, weight=1)

    def apply_profile_name(self) -> None:
        name = self.user_name.get().strip() or "User"
        self.user_name.set(name)
        self.add_meta_message(f"Profile updated: {name}")

    def use_command(self, command: str) -> None:
        if command == "/clear":
            self.clear_chat()
            return
        if command == "/save":
            self.save_chat()
            return
        if command.startswith("/name"):
            self.name_entry.focus_set()
            self.name_entry.select_range(0, tk.END)
            return
        if "<query>" in command:
            self.user_input.delete(0, tk.END)
            self.user_input.insert(0, "/search ")
            self.user_input.focus_set()
            return
        if "<prompt>" in command:
            self.user_input.delete(0, tk.END)
            self.user_input.insert(0, "/image ")
            self.user_input.focus_set()
            return
        if "<question>" in command:
            self.user_input.delete(0, tk.END)
            self.user_input.insert(0, "/analyze ")
            self.user_input.focus_set()
            return
        self.send_message(command)

    def send_message(self, preset_message: str | None = None) -> None:
        if self.engine is None:
            messagebox.showwarning("Chatbot Not Ready", "The chatbot engine did not load correctly.")
            return

        message = preset_message if preset_message is not None else self.user_input.get()
        message = message.strip()
        self.user_input.delete(0, tk.END)

        if not message:
            return

        if message.lower().startswith("/name "):
            new_name = message.split(" ", 1)[1].strip()
            if new_name:
                self.user_name.set(new_name)
                self.add_user_message(message)
                self.add_bot_message(f"Done. I will call you {new_name}.")
                self.conversation_history.append((message, f"Done. I will call you {new_name}."))
            return

        if message.lower() == "/clear":
            self.clear_chat()
            return

        if message.lower() == "/save":
            self.save_chat()
            return

        self.add_user_message(message)
        reply = self.engine.reply(message, user_name=self.user_name.get().strip() or "User")
        self.add_bot_message(reply.text)
        self.conversation_history.append((message, reply.text))

        if reply.action == "open_url" and reply.url:
            webbrowser.open(reply.url)
        elif reply.action == "generate_image" and reply.prompt:
            self.start_image_generation(reply.prompt)
        elif reply.action == "analyze_image":
            self.choose_and_analyze_image(reply.prompt or "")

    def generate_image_from_input(self) -> None:
        prompt = self.user_input.get().strip()
        if prompt.lower().startswith("/image "):
            self.send_message(prompt)
            return
        if not prompt:
            messagebox.showinfo("Describe an Image", "Type an image description in the message box first.")
            self.user_input.focus_set()
            return
        self.send_message(f"/image {prompt}")

    def get_ai_service(self) -> AIImageService:
        if self.ai_service is None:
            self.ai_service = AIImageService()
        return self.ai_service

    def run_ai_task(self, status: str, worker, on_success) -> None:
        if self.ai_busy:
            messagebox.showinfo("AI Image Tool", "Wait for the current AI image task to finish.")
            return
        self.ai_busy = True
        for button in self.ai_buttons:
            button.configure(state=tk.DISABLED)
        self.status_text.set(status)

        def run() -> None:
            try:
                result = worker()
            except Exception as exc:
                error_message = str(exc)
                self.root.after(0, lambda: self.finish_ai_task(error=error_message))
                return
            self.root.after(0, lambda: self.finish_ai_task(result=result, on_success=on_success))

        threading.Thread(target=run, daemon=True).start()

    def finish_ai_task(self, result=None, on_success=None, error: str | None = None) -> None:
        self.ai_busy = False
        for button in self.ai_buttons:
            button.configure(state=tk.NORMAL)
        self.status_text.set("Ready. Choose a command or ask a question.")
        if error:
            self.add_bot_message(error)
            messagebox.showerror("AI Image Tool", error)
            return
        if on_success:
            on_success(result)

    def start_image_generation(self, prompt: str) -> None:
        self.run_ai_task(
            "Generating your image...",
            lambda: self.get_ai_service().generate_image(prompt),
            self.show_generated_image,
        )

    def show_generated_image(self, generated_image) -> None:
        message = f"{generated_image.message}\nSaved to: {generated_image.path}"
        self.add_bot_message(message)
        self.conversation_history.append(("[Generated image]", message))
        try:
            image = tk.PhotoImage(file=str(generated_image.path))
            scale = max(1, (max(image.width(), image.height()) + 419) // 420)
            if scale > 1:
                image = image.subsample(scale, scale)
            self.generated_image_refs.append(image)
            self.chat_history.configure(state=tk.NORMAL)
            self.chat_history.image_create(tk.END, image=image)
            self.chat_history.insert(tk.END, "\n\n")
            self.chat_history.see(tk.END)
            self.chat_history.configure(state=tk.DISABLED)
        except tk.TclError:
            self.add_meta_message("Preview is unavailable, but the generated image was saved.")

    def choose_and_analyze_image(self, question: str = "") -> None:
        image_path = filedialog.askopenfilename(
            title="Choose an image to predict",
            filetypes=[
                ("Supported images", "*.png *.jpg *.jpeg *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not image_path:
            self.add_meta_message("Image prediction cancelled.")
            return

        selected_path = Path(image_path)
        self.add_meta_message(f"Selected device image: {selected_path.name}")
        self.run_ai_task(
            "Predicting the uploaded image...",
            lambda: self.get_ai_service().analyze_image(selected_path, question),
            lambda prediction: self.show_image_prediction(selected_path, prediction),
        )

    def show_image_prediction(self, image_path: Path, prediction: str) -> None:
        message = f"Prediction for {image_path.name}:\n{prediction}"
        self.add_bot_message(message)
        self.conversation_history.append((f"[Device image] {image_path.name}", prediction))

    def add_user_message(self, message: str) -> None:
        self.append_chat("You", message, "user")

    def add_bot_message(self, message: str) -> None:
        self.append_chat("Bot", message, "bot")

    def add_meta_message(self, message: str) -> None:
        timestamp = dt.datetime.now().strftime("%I:%M %p")
        self.chat_history.configure(state=tk.NORMAL)
        self.chat_history.insert(tk.END, f"{timestamp}  {message}\n\n", "meta")
        self.chat_history.see(tk.END)
        self.chat_history.configure(state=tk.DISABLED)

    def append_chat(self, speaker: str, message: str, tag: str) -> None:
        timestamp = dt.datetime.now().strftime("%I:%M %p")
        self.chat_history.configure(state=tk.NORMAL)
        self.chat_history.insert(tk.END, f"{speaker} ", tag)
        self.chat_history.insert(tk.END, f"{timestamp}\n", "meta")
        self.chat_history.insert(tk.END, f"{message}\n\n")
        self.chat_history.see(tk.END)
        self.chat_history.configure(state=tk.DISABLED)

    def clear_chat(self) -> None:
        self.chat_history.configure(state=tk.NORMAL)
        self.chat_history.delete("1.0", tk.END)
        self.chat_history.configure(state=tk.DISABLED)
        self.conversation_history.clear()
        self.add_meta_message("Conversation cleared.")

    def save_chat(self) -> None:
        if not self.conversation_history:
            messagebox.showinfo("Nothing to Save", "There is no conversation to save yet.")
            return

        default_name = f"chat_history_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        save_path = filedialog.asksaveasfilename(
            initialdir=APP_DIR,
            initialfile=default_name,
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not save_path:
            return

        with Path(save_path).open("w", encoding="utf-8") as file:
            file.write(f"User profile: {self.user_name.get().strip() or 'User'}\n")
            file.write(f"Saved: {dt.datetime.now().isoformat(timespec='seconds')}\n\n")
            for user_message, bot_message in self.conversation_history:
                file.write(f"You: {user_message}\n")
                file.write(f"Bot: {bot_message}\n\n")

        self.add_meta_message(f"Conversation saved to {save_path}")


if __name__ == "__main__":
    window = tk.Tk()
    ChatbotProfileUI(window)
    window.mainloop()

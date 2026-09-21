import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk

from pathlib import Path
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText

import torch

from amber.model import AmberConfig, AmberModel
from amber.updater import open_updater_window
from inference.chat import generate_reply


ROOT = Path(__file__).resolve().parent

CHECKPOINTS = (
    ROOT
    / "checkpoints"
)

LATEST_CHECKPOINT = (
    CHECKPOINTS
    / "amber_seed_latest.pt"
)

STOP_FILE = (
    ROOT
    / "training"
    / ".stop_requested"
)

PAUSE_FILE = (
    ROOT
    / "training"
    / ".pause_requested"
)


class AmberApp(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title(
            "Amber 0.0.4"
        )

        self.geometry(
            "1180x760"
        )

        self.minsize(
            1000,
            650
        )

        self.configure(
            bg="#101014"
        )

        self.process = None
        self.log_queue = queue.Queue()

        self.mode = tk.StringVar(
            value="BALANCED"
        )

        self.target_step = tk.StringVar(
            value="500"
        )

        self.current_step = 0
        self.current_loss = None

        self.paused = False

        self._style()
        self._build_ui()

        self.after(
            200,
            self._poll_logs
        )

        self.after(
            500,
            self.refresh_status
        )

    # ========================================================
    # STYLE
    # ========================================================

    def _style(self):
        style = ttk.Style(
            self
        )

        try:
            style.theme_use(
                "clam"
            )
        except Exception:
            pass

        style.configure(
            ".",
            background="#101014",
            foreground="#ffffff"
        )

        style.configure(
            "TFrame",
            background="#101014"
        )

        style.configure(
            "Card.TFrame",
            background="#191920"
        )

        style.configure(
            "TLabel",
            background="#101014",
            foreground="#ffffff",
            font=("Segoe UI", 10)
        )

        style.configure(
            "Title.TLabel",
            background="#101014",
            foreground="#ffffff",
            font=(
                "Segoe UI Semibold",
                25
            )
        )

        style.configure(
            "Subtitle.TLabel",
            background="#101014",
            foreground="#9999a6",
            font=("Segoe UI", 10)
        )

        style.configure(
            "CardTitle.TLabel",
            background="#191920",
            foreground="#a5a5b0",
            font=(
                "Segoe UI Semibold",
                9
            )
        )

        style.configure(
            "CardValue.TLabel",
            background="#191920",
            foreground="#ffb347",
            font=(
                "Segoe UI Semibold",
                14
            )
        )

        style.configure(
            "TButton",
            font=(
                "Segoe UI Semibold",
                10
            ),
            padding=10
        )

        style.configure(
            "TRadiobutton",
            background="#101014",
            foreground="#ffffff"
        )

        # Champs de saisie Amber
        style.configure(
            "TEntry",
            fieldbackground="#191920",
            foreground="#ffffff",
            bordercolor="#34343d",
            lightcolor="#34343d",
            darkcolor="#34343d",
            padding=6
        )

        style.map(
            "TEntry",
            fieldbackground=[
                ("focus", "#202028")
            ],
            foreground=[
                ("disabled", "#777780"),
                ("!disabled", "#ffffff")
            ]
        )

        style.configure(
            "TNotebook",
            background="#101014",
            borderwidth=0
        )

        style.configure(
            "TNotebook.Tab",
            background="#24242c",
            foreground="#b5b5bf",
            padding=[
                20,
                10
            ],
            font=(
                "Segoe UI Semibold",
                10
            )
        )

        style.map(
            "TNotebook.Tab",
            background=[
                ("selected", "#34343e"),
                ("active", "#2b2b34")
            ],
            foreground=[
                ("selected", "#ffb347"),
                ("active", "#ffffff")
            ]
        )

    # ========================================================
    # UI
    # ========================================================

    def _build_ui(self):

        root = ttk.Frame(
            self
        )

        root.pack(
            fill="both",
            expand=True,
            padx=24,
            pady=18
        )

        ttk.Label(
            root,
            text="AMBER",
            style="Title.TLabel"
        ).pack(
            anchor="w"
        )

        ttk.Label(
            root,
            text="AI Control Center · Amber Model 0.0.4",
            style="Subtitle.TLabel"
        ).pack(
            anchor="w",
            pady=(0, 14)
        )

        self.notebook = ttk.Notebook(
            root
        )

        self.notebook.pack(
            fill="both",
            expand=True
        )

        self.dashboard_tab = ttk.Frame(
            self.notebook
        )

        self.chat_tab = ttk.Frame(
            self.notebook
        )

        self.training_tab = ttk.Frame(
            self.notebook
        )

        self.checkpoint_tab = ttk.Frame(
            self.notebook
        )

        self.notebook.add(
            self.dashboard_tab,
            text="Dashboard"
        )

        self.notebook.add(
            self.chat_tab,
            text="Chat"
        )

        self.notebook.add(
            self.training_tab,
            text="Training"
        )

        self.notebook.add(
            self.checkpoint_tab,
            text="Checkpoints"
        )

        self._build_dashboard()
        self._build_chat()
        self._build_training()
        self._build_checkpoints()

    # ========================================================
    # DASHBOARD
    # ========================================================

    def _build_dashboard(self):

        cards = ttk.Frame(
            self.dashboard_tab
        )

        cards.pack(
            fill="x",
            pady=20
        )

        self.model_value = self._card(
            cards,
            "MODEL"
        )

        self.gpu_value = self._card(
            cards,
            "GPU"
        )

        self.vram_value = self._card(
            cards,
            "VRAM"
        )

        self.step_value = self._card(
            cards,
            "STEP"
        )

        self.loss_value = self._card(
            cards,
            "LOSS"
        )

        buttons = ttk.Frame(
            self.dashboard_tab
        )

        buttons.pack(
            anchor="w",
            pady=10
        )

        ttk.Button(
            buttons,
            text="Tester Amber",
            command=self.test_brain
        ).pack(
            side="left",
            padx=(0, 8)
        )

        ttk.Button(
            buttons,
            text="Actualiser",
            command=self.refresh_status
        ).pack(
            side="left",
            padx=8
        )

        ttk.Button(
            buttons,
            text="Ouvrir C:\\Amber",
            command=lambda: os.startfile(
                ROOT
            )
        ).pack(
            side="left",
            padx=8
        )

        ttk.Button(
            buttons,
            text="Mises à jour",
            command=lambda: open_updater_window(
                self,
                process_getter=lambda: self.process
            )
        ).pack(
            side="left",
            padx=8
        )

        ttk.Label(
            self.dashboard_tab,
            text="Console Amber"
        ).pack(
            anchor="w",
            pady=(20, 5)
        )

        self.console = ScrolledText(
            self.dashboard_tab,
            bg="#0b0b0f",
            fg="#e5e5ea",
            insertbackground="white",
            relief="flat",
            font=(
                "Consolas",
                10
            ),
            height=20
        )

        self.console.pack(
            fill="both",
            expand=True
        )

        self.log(
            "Amber Control Center 0.0.3 ready."
        )

    def _card(
        self,
        parent,
        title
    ):
        frame = ttk.Frame(
            parent,
            style="Card.TFrame"
        )

        frame.pack(
            side="left",
            fill="both",
            expand=True,
            padx=5
        )

        ttk.Label(
            frame,
            text=title,
            style="CardTitle.TLabel"
        ).pack(
            anchor="w",
            padx=14,
            pady=(12, 2)
        )

        value = ttk.Label(
            frame,
            text="...",
            style="CardValue.TLabel"
        )

        value.pack(
            anchor="w",
            padx=14,
            pady=(0, 12)
        )

        return value

    # ========================================================
    # CHAT
    # ========================================================

    def _build_chat(self):

        ttk.Label(
            self.chat_tab,
            text="Chat avec Amber Seed",
            style="Title.TLabel"
        ).pack(
            anchor="w",
            pady=(20, 5)
        )

        ttk.Label(
            self.chat_tab,
            text=(
                "Réponses générées uniquement par le checkpoint "
                "local Amber."
            ),
            style="Subtitle.TLabel"
        ).pack(
            anchor="w",
            pady=(0, 12)
        )

        self.chat_history = ScrolledText(
            self.chat_tab,
            bg="#0b0b0f",
            fg="#eeeeee",
            insertbackground="white",
            relief="flat",
            font=(
                "Segoe UI",
                11
            ),
            state="disabled"
        )

        self.chat_history.pack(
            fill="both",
            expand=True,
            pady=(0, 10)
        )

        self.chat_input = tk.Text(
            self.chat_tab,
            height=4,
            bg="#191920",
            fg="#ffffff",
            insertbackground="white",
            relief="flat",
            font=(
                "Segoe UI",
                11
            )
        )

        self.chat_input.pack(
            fill="x"
        )

        actions = ttk.Frame(
            self.chat_tab
        )

        actions.pack(
            fill="x",
            pady=10
        )

        self.send_button = ttk.Button(
            actions,
            text="Envoyer à Amber",
            command=self.send_chat
        )

        self.send_button.pack(
            side="right"
        )

    def add_chat(
        self,
        author,
        text
    ):
        self.chat_history.configure(
            state="normal"
        )

        self.chat_history.insert(
            "end",
            f"{author}\n{text}\n\n"
        )

        self.chat_history.configure(
            state="disabled"
        )

        self.chat_history.see(
            "end"
        )

    def send_chat(self):

        if (
            self.process is not None
            and self.process.poll() is None
        ):
            messagebox.showwarning(
                "Amber",
                "Arrête ou mets fin à l'entraînement avant d'utiliser le Chat."
            )

            return

        text = self.chat_input.get(
            "1.0",
            "end"
        ).strip()

        if not text:
            return

        if not LATEST_CHECKPOINT.exists():
            messagebox.showwarning(
                "Amber",
                "Aucun checkpoint disponible."
            )

            return

        self.chat_input.delete(
            "1.0",
            "end"
        )

        self.add_chat(
            "Vous",
            text
        )

        self.add_chat(
            "Amber",
            "..."
        )

        self.send_button.config(
            state="disabled"
        )

        def worker():
            try:
                result = generate_reply(
                    text
                )

                self.after(
                    0,
                    lambda: self._finish_chat(
                        result
                    )
                )

            except Exception as exc:
                self.after(
                    0,
                    lambda: self._chat_error(
                        str(exc)
                    )
                )

        threading.Thread(
            target=worker,
            daemon=True
        ).start()

    def _remove_last_waiting_message(
        self
    ):
        self.chat_history.configure(
            state="normal"
        )

        content = self.chat_history.get(
            "1.0",
            "end"
        )

        marker = "Amber\n...\n\n"

        index = content.rfind(
            marker
        )

        if index >= 0:
            new_content = (
                content[:index]
                + content[
                    index + len(marker):
                ]
            )

            self.chat_history.delete(
                "1.0",
                "end"
            )

            self.chat_history.insert(
                "1.0",
                new_content
            )

        self.chat_history.configure(
            state="disabled"
        )

    def _finish_chat(
        self,
        result
    ):
        self._remove_last_waiting_message()

        answer = result.get(
            "text",
            ""
        )

        if not answer:
            answer = (
                "[Amber n'a pas produit de texte lisible]"
            )

        self.add_chat(
            "Amber",
            answer
        )

        self.send_button.config(
            state="normal"
        )

    def _chat_error(
        self,
        error
    ):
        self._remove_last_waiting_message()

        self.add_chat(
            "Amber",
            f"[Erreur : {error}]"
        )

        self.send_button.config(
            state="normal"
        )

    # ========================================================
    # TRAINING
    # ========================================================

    def _build_training(self):

        top = ttk.Frame(
            self.training_tab
        )

        top.pack(
            fill="x",
            pady=20
        )

        ttk.Label(
            top,
            text="Training Governor"
        ).pack(
            anchor="w"
        )

        modes = ttk.Frame(
            top
        )

        modes.pack(
            anchor="w",
            pady=10
        )

        for mode in [
            "ECO",
            "BALANCED",
            "FULL"
        ]:
            ttk.Radiobutton(
                modes,
                text=mode,
                variable=self.mode,
                value=mode
            ).pack(
                side="left",
                padx=(0, 20)
            )

        objective = ttk.Frame(
            self.training_tab
        )

        objective.pack(
            anchor="w",
            pady=10
        )

        ttk.Label(
            objective,
            text="Objectif étape :"
        ).pack(
            side="left"
        )

        ttk.Entry(
            objective,
            textvariable=self.target_step,
            width=12
        ).pack(
            side="left",
            padx=10
        )

        self.training_status = ttk.Label(
            self.training_tab,
            text="Prêt"
        )

        self.training_status.pack(
            anchor="w",
            pady=(10, 5)
        )

        self.progress = ttk.Progressbar(
            self.training_tab,
            orient="horizontal",
            mode="determinate"
        )

        self.progress.pack(
            fill="x",
            pady=5
        )

        self.progress_text = ttk.Label(
            self.training_tab,
            text="0 / 500"
        )

        self.progress_text.pack(
            anchor="w"
        )

        buttons = ttk.Frame(
            self.training_tab
        )

        buttons.pack(
            anchor="w",
            pady=20
        )

        ttk.Button(
            buttons,
            text="Démarrer",
            command=self.start_training
        ).pack(
            side="left",
            padx=(0, 8)
        )

        self.pause_button = ttk.Button(
            buttons,
            text="Pause",
            command=self.toggle_pause
        )

        self.pause_button.pack(
            side="left",
            padx=8
        )

        ttk.Button(
            buttons,
            text="Sauvegarder et arrêter",
            command=self.stop_process
        ).pack(
            side="left",
            padx=8
        )

        ttk.Button(
            buttons,
            text="Checkpoints",
            command=lambda: os.startfile(
                CHECKPOINTS
            )
        ).pack(
            side="left",
            padx=8
        )

    def start_training(self):

        if (
            self.process is not None
            and self.process.poll() is None
        ):
            messagebox.showwarning(
                "Amber",
                "Une tâche Amber est déjà en cours."
            )

            return

        try:
            target = int(
                self.target_step.get()
            )

        except ValueError:
            messagebox.showerror(
                "Amber",
                "L'objectif doit être un nombre entier."
            )

            return

        if target <= self.current_step:
            messagebox.showwarning(
                "Amber",
                (
                    f"Amber est déjà à l'étape "
                    f"{self.current_step}."
                )
            )

            return

        try:
            if STOP_FILE.exists():
                STOP_FILE.unlink()

            if PAUSE_FILE.exists():
                PAUSE_FILE.unlink()

        except Exception:
            pass

        self.paused = False

        self.pause_button.config(
            text="Pause"
        )

        self.progress.configure(
            maximum=target,
            value=self.current_step
        )

        self.progress_text.config(
            text=f"{self.current_step} / {target}"
        )

        self.training_status.config(
            text="Entraînement en cours..."
        )

        self.notebook.select(
            self.dashboard_tab
        )

        self.log("")
        self.log(
            f"=== TRAINING {self.mode.get()} ==="
        )

        self._run_command(
            [
                sys.executable,
                "-m",
                "training.train",
                "--mode",
                self.mode.get().lower(),
                "--target-step",
                str(target)
            ]
        )

    def toggle_pause(self):

        if (
            self.process is None
            or self.process.poll() is not None
        ):
            return

        try:
            if not self.paused:

                PAUSE_FILE.write_text(
                    "pause",
                    encoding="utf-8"
                )

                self.paused = True

                self.pause_button.config(
                    text="Reprendre"
                )

                self.training_status.config(
                    text="Pause demandée..."
                )

            else:

                if PAUSE_FILE.exists():
                    PAUSE_FILE.unlink()

                self.paused = False

                self.pause_button.config(
                    text="Pause"
                )

                self.training_status.config(
                    text="Reprise..."
                )

        except Exception as exc:

            messagebox.showerror(
                "Amber",
                str(exc)
            )

    def stop_process(self):

        if (
            self.process is None
            or self.process.poll() is not None
        ):
            self.log(
                "Aucun entraînement Amber actif."
            )

            return

        try:
            STOP_FILE.write_text(
                "stop",
                encoding="utf-8"
            )

            self.training_status.config(
                text="Sauvegarde et arrêt..."
            )

            self.log("")
            self.log(
                "Clean stop requested..."
            )

        except Exception as exc:

            self.log(
                f"Stop error: {exc}"
            )

    # ========================================================
    # CHECKPOINTS
    # ========================================================

    def _build_checkpoints(self):

        ttk.Label(
            self.checkpoint_tab,
            text="Checkpoints Amber",
            style="Title.TLabel"
        ).pack(
            anchor="w",
            pady=(20, 10)
        )

        self.checkpoint_list = tk.Listbox(
            self.checkpoint_tab,
            bg="#0b0b0f",
            fg="#ffffff",
            relief="flat",
            font=(
                "Consolas",
                10
            )
        )

        self.checkpoint_list.pack(
            fill="both",
            expand=True
        )

        controls = ttk.Frame(
            self.checkpoint_tab
        )

        controls.pack(
            fill="x",
            pady=10
        )

        ttk.Button(
            controls,
            text="Actualiser",
            command=self.refresh_checkpoints
        ).pack(
            side="left",
            padx=(0, 8)
        )

        ttk.Button(
            controls,
            text="Ouvrir le dossier",
            command=lambda: os.startfile(
                CHECKPOINTS
            )
        ).pack(
            side="left"
        )

    def refresh_checkpoints(self):

        CHECKPOINTS.mkdir(
            parents=True,
            exist_ok=True
        )

        self.checkpoint_list.delete(
            0,
            "end"
        )

        files = sorted(
            CHECKPOINTS.glob(
                "*.pt"
            ),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        if not files:
            self.checkpoint_list.insert(
                "end",
                "Aucun checkpoint."
            )

            return

        for file in files:

            size_mb = (
                file.stat().st_size
                / 1024**2
            )

            self.checkpoint_list.insert(
                "end",
                (
                    f"{file.name}   "
                    f"{size_mb:.1f} MB"
                )
            )

    # ========================================================
    # STATUS
    # ========================================================

    def find_rx7900xt(self):

        if not torch.cuda.is_available():
            return None

        for i in range(
            torch.cuda.device_count()
        ):
            name = torch.cuda.get_device_name(
                i
            )

            if "7900 XT" in name.upper():
                return i

        return None

    def checkpoint_info(self):

        if not LATEST_CHECKPOINT.exists():
            return {
                "step": 0,
                "loss": None
            }

        try:
            checkpoint = torch.load(
                LATEST_CHECKPOINT,
                map_location="cpu",
                weights_only=False
            )

            return {
                "step": int(
                    checkpoint.get(
                        "step",
                        0
                    )
                ),
                "loss": checkpoint.get(
                    "loss",
                    None
                )
            }

        except Exception:
            return {
                "step": 0,
                "loss": None
            }

    def refresh_status(self):

        config = AmberConfig()

        model = AmberModel(
            config
        )

        parameters = model.parameter_count()

        self.model_value.config(
            text=(
                f"{parameters / 1_000_000:.2f} M"
            )
        )

        del model

        gpu_id = self.find_rx7900xt()

        if gpu_id is not None:

            name = torch.cuda.get_device_name(
                gpu_id
            )

            properties = torch.cuda.get_device_properties(
                gpu_id
            )

            vram = (
                properties.total_memory
                / 1024**3
            )

            self.gpu_value.config(
                text="RX 7900 XT"
            )

            self.vram_value.config(
                text=f"{vram:.1f} GB"
            )

        else:

            self.gpu_value.config(
                text="Not found"
            )

            self.vram_value.config(
                text="-"
            )

        info = self.checkpoint_info()

        self.current_step = info[
            "step"
        ]

        self.current_loss = info[
            "loss"
        ]

        self.step_value.config(
            text=str(
                self.current_step
            )
        )

        if self.current_loss is None:

            self.loss_value.config(
                text="-"
            )

        else:

            self.loss_value.config(
                text=f"{float(self.current_loss):.4f}"
            )

        try:
            target = int(
                self.target_step.get()
            )

            self.progress.configure(
                maximum=max(
                    target,
                    1
                ),
                value=min(
                    self.current_step,
                    target
                )
            )

            self.progress_text.config(
                text=(
                    f"{self.current_step} / "
                    f"{target}"
                )
            )

        except Exception:
            pass

        self.refresh_checkpoints()

    # ========================================================
    # PROCESS / LOGS
    # ========================================================

    def log(
        self,
        text
    ):
        self.console.insert(
            "end",
            text + "\n"
        )

        self.console.see(
            "end"
        )

    def _run_command(
        self,
        args
    ):

        if (
            self.process is not None
            and self.process.poll() is None
        ):
            return

        def worker():
            try:

                env = os.environ.copy()

                env[
                    "PYTHONUTF8"
                ] = "1"

                env[
                    "PYTHONIOENCODING"
                ] = "utf-8"

                creation_flags = 0

                if os.name == "nt":
                    creation_flags = (
                        subprocess.CREATE_NO_WINDOW
                    )

                self.process = subprocess.Popen(
                    args,
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                    creationflags=creation_flags
                )

                for line in self.process.stdout:

                    self.log_queue.put(
                        line.rstrip()
                    )

                code = self.process.wait()

                self.log_queue.put(
                    f"[Process finished: {code}]"
                )

            except Exception as exc:

                self.log_queue.put(
                    f"[ERROR] {exc}"
                )

            finally:

                self.process = None

        threading.Thread(
            target=worker,
            daemon=True
        ).start()

    def _poll_logs(self):

        while True:

            try:
                line = self.log_queue.get_nowait()

            except queue.Empty:
                break

            self.log(
                line
            )

            match = re.search(
                r"\[(\d{6})\]\s+loss=([0-9.]+)",
                line
            )

            if match:

                step = int(
                    match.group(1)
                )

                loss = float(
                    match.group(2)
                )

                self.current_step = step
                self.current_loss = loss

                self.step_value.config(
                    text=str(
                        step
                    )
                )

                self.loss_value.config(
                    text=f"{loss:.4f}"
                )

                try:

                    target = int(
                        self.target_step.get()
                    )

                    self.progress.configure(
                        maximum=target,
                        value=min(
                            step,
                            target
                        )
                    )

                    self.progress_text.config(
                        text=f"{step} / {target}"
                    )

                except Exception:
                    pass

            if (
                "[Checkpoint]"
                in line
            ):
                self.refresh_checkpoints()

            if (
                "Training paused"
                in line
            ):
                self.training_status.config(
                    text="En pause"
                )

            if (
                "Training resumed"
                in line
            ):
                self.training_status.config(
                    text="Entraînement en cours..."
                )

            if (
                "Clean stop completed"
                in line
            ):
                self.training_status.config(
                    text="Arrêté proprement"
                )

            if (
                "AMBER TRAINING COMPLETE"
                in line
            ):
                self.training_status.config(
                    text="Objectif atteint"
                )

            if (
                "[Process finished:"
                in line
            ):
                self.after(
                    300,
                    self.refresh_status
                )

                self.paused = False

                self.pause_button.config(
                    text="Pause"
                )

        self.after(
            200,
            self._poll_logs
        )

    # ========================================================
    # TEST
    # ========================================================

    def test_brain(self):

        self.log("")
        self.log(
            "=== AMBER BRAIN TEST ==="
        )

        self._run_command(
            [
                sys.executable,
                "-m",
                "tests.test_brain"
            ]
        )


if __name__ == "__main__":
    app = AmberApp()
    app.mainloop()

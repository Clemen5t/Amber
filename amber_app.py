import os
import json
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk

from pathlib import Path
from datetime import datetime
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText

import torch

from amber.model import AmberConfig, AmberModel
from amber.model_v01 import (
    AmberV01Config,
    estimate_v01_parameter_count,
)
from training.data_v01 import (
    CACHE_META as V01_CACHE_META,
    load_cache_metadata as load_v01_cache_metadata,
)
from amber.dataset_manager import (
    SOURCE_CATALOG,
    PREPARED_TRAIN_FILE,
    PREPARED_VALIDATION_FILE,
    MANIFEST_FILE,
    DownloadCancelled,
    dataset_stats,
    file_stats,
    format_bytes,
    format_tokens,
    import_local_file,
    download_source,
    download_catalog_sources,
    estimate_source_plan,
    list_raw_sources,
    prepare_dataset,
    write_manifest,
)
from tokenizer.train_tokenizer import (
    train_amber_tokenizer,
    tokenizer_status,
)
from amber.updater import open_updater_window
from inference.chat import generate_reply


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

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

V01_STOP_FILE = (
    ROOT
    / "training"
    / ".v01_stop_requested"
)

V01_PAUSE_FILE = (
    ROOT
    / "training"
    / ".v01_pause_requested"
)

V01_STATUS_FILE = (
    CHECKPOINTS
    / "amber_v01_status.json"
)

V01_CHECKPOINT = (
    CHECKPOINTS
    / "amber_v01_latest.pt"
)

V01_AUTOTUNE_FILE = (
    CHECKPOINTS
    / "amber_v01_autotune.json"
)


class AmberApp(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title(
            "Amber 0.1.7"
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

        self.v01_mode = tk.StringVar(
            value="BALANCED"
        )

        self.v01_target_tokens = tk.StringVar(
            value="100000000"
        )

        self.v01_schedule_tokens = tk.StringVar(
            value="100000000"
        )

        self.v01_full_vram_target = tk.StringVar(
            value="15"
        )

        self.v01_gen_mode = tk.StringVar(
            value="Continuation"
        )

        self.v01_gen_temperature = tk.StringVar(
            value="0.8"
        )

        self.v01_gen_top_k = tk.StringVar(
            value="50"
        )

        self.v01_gen_top_p = tk.StringVar(
            value="0.95"
        )

        self.v01_gen_max_tokens = tk.StringVar(
            value="120"
        )

        self.v01_gen_repetition = tk.StringVar(
            value="1.05"
        )

        self.v01_eval_batches = tk.StringVar(
            value="50"
        )

        self.current_step = 0
        self.current_loss = None
        self.current_speed = 0.0
        self.current_eta = "--:--"
        self.current_train_vram = 0.0
        self.current_governor = "BALANCED"

        self.paused = False
        self.v01_paused = False
        self.dataset_busy = False
        self.dataset_cancel_event = threading.Event()

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
            text="AI Control Center · Amber Model 0.1.7",
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

        self.dataset_tab = ttk.Frame(
            self.notebook
        )

        self.v01_tab = ttk.Frame(
            self.notebook
        )

        self.evaluation_tab = ttk.Frame(
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

        self.notebook.add(
            self.dataset_tab,
            text="Dataset"
        )

        self.notebook.add(
            self.v01_tab,
            text="Amber 0.1"
        )

        self.notebook.add(
            self.evaluation_tab,
            text="Évaluation 0.1"
        )

        self._build_dashboard()
        self._build_chat()
        self._build_training()
        self._build_checkpoints()
        self._build_dataset()
        self._build_v01()
        self._build_v01_evaluation()

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
            "Amber Control Center 0.1.7 ready."
        )

    def _timestamp(self):
        return datetime.now().strftime(
            "%H:%M:%S"
        )

    def _timestamped_line(
        self,
        text
    ):
        if text == "":
            return "\n"

        return (
            f"[{self._timestamp()}] "
            f"{text}\n"
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

        self.chat_input.bind(
            "<Control-Return>",
            lambda event: (
                self.send_chat(),
                "break"
            )[1]
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
    # AMBER 0.1 - EVALUATION & GENERATION
    # ========================================================

    def _build_v01_evaluation(self):

        ttk.Label(
            self.evaluation_tab,
            text="Amber 0.1 · Évaluation & génération",
            style="Title.TLabel"
        ).pack(
            anchor="w",
            pady=(18, 4)
        )

        ttk.Label(
            self.evaluation_tab,
            text=(
                "Teste le checkpoint pré-entraîné. "
                "Amber 0.1 est un modèle de complétion : "
                "le mode Question-Réponse reste expérimental tant "
                "qu'un post-entraînement instruction n'a pas été fait."
            ),
            style="Subtitle.TLabel"
        ).pack(
            anchor="w",
            pady=(0, 10)
        )

        cards = ttk.Frame(
            self.evaluation_tab
        )

        cards.pack(
            fill="x",
            pady=(0, 10)
        )

        self.eval_checkpoint_value = self._card(
            cards,
            "TOKENS ENTRAÎNÉS"
        )

        self.eval_stored_val_value = self._card(
            cards,
            "VAL CHECKPOINT"
        )

        self.eval_loss_value = self._card(
            cards,
            "VAL MESURÉE"
        )

        self.eval_ppl_value = self._card(
            cards,
            "PERPLEXITÉ"
        )

        top = ttk.Frame(
            self.evaluation_tab
        )

        top.pack(
            fill="both",
            expand=True
        )

        left = ttk.Frame(
            top
        )

        left.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(0, 8)
        )

        right = ttk.Frame(
            top
        )

        right.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(8, 0)
        )

        ttk.Label(
            left,
            text="Prompt"
        ).pack(
            anchor="w",
            pady=(0, 4)
        )

        self.v01_prompt_input = ScrolledText(
            left,
            bg="#191920",
            fg="#ffffff",
            insertbackground="white",
            relief="flat",
            font=("Segoe UI", 11),
            height=12
        )

        self.v01_prompt_input.pack(
            fill="both",
            expand=True
        )

        self.v01_prompt_input.insert(
            "1.0",
            "La France est un pays"
        )

        ttk.Label(
            right,
            text="Sortie Amber 0.1"
        ).pack(
            anchor="w",
            pady=(0, 4)
        )

        self.v01_generation_output = ScrolledText(
            right,
            bg="#0b0b0f",
            fg="#eeeeee",
            insertbackground="white",
            relief="flat",
            font=("Segoe UI", 11),
            height=12
        )

        self.v01_generation_output.pack(
            fill="both",
            expand=True
        )

        generation_controls = ttk.Frame(
            self.evaluation_tab
        )

        generation_controls.pack(
            fill="x",
            pady=(8, 4)
        )

        ttk.Label(
            generation_controls,
            text="Mode :"
        ).pack(
            side="left"
        )

        ttk.Combobox(
            generation_controls,
            textvariable=self.v01_gen_mode,
            values=[
                "Continuation",
                "Question-Réponse expérimental"
            ],
            state="readonly",
            width=27
        ).pack(
            side="left",
            padx=(5, 12)
        )

        for label, variable, width in [
            ("Temp", self.v01_gen_temperature, 6),
            ("Top-k", self.v01_gen_top_k, 6),
            ("Top-p", self.v01_gen_top_p, 6),
            ("Tokens", self.v01_gen_max_tokens, 7),
            ("Répét.", self.v01_gen_repetition, 6),
        ]:
            ttk.Label(
                generation_controls,
                text=label + " :"
            ).pack(
                side="left",
                padx=(4, 2)
            )

            ttk.Entry(
                generation_controls,
                textvariable=variable,
                width=width
            ).pack(
                side="left",
                padx=(0, 5)
            )

        ttk.Button(
            generation_controls,
            text="Générer",
            command=self.run_v01_generation
        ).pack(
            side="left",
            padx=(12, 5)
        )

        ttk.Button(
            generation_controls,
            text="Effacer",
            command=lambda: self.v01_generation_output.delete(
                "1.0",
                "end"
            )
        ).pack(
            side="left",
            padx=5
        )

        eval_controls = ttk.Frame(
            self.evaluation_tab
        )

        eval_controls.pack(
            fill="x",
            pady=(4, 8)
        )

        ttk.Label(
            eval_controls,
            text="Batches validation :"
        ).pack(
            side="left"
        )

        ttk.Entry(
            eval_controls,
            textvariable=self.v01_eval_batches,
            width=8
        ).pack(
            side="left",
            padx=6
        )

        ttk.Button(
            eval_controls,
            text="Mesurer loss + perplexité",
            command=self.run_v01_evaluation
        ).pack(
            side="left",
            padx=6
        )

        self.v01_eval_status = ttk.Label(
            eval_controls,
            text="Prêt"
        )

        self.v01_eval_status.pack(
            side="left",
            padx=14
        )

        self.refresh_v01_eval_status()

    def refresh_v01_eval_status(self):

        status = self._read_v01_status_file()

        tokens_seen = int(
            status.get(
                "tokens_seen",
                0
            )
        )

        stored_val = status.get(
            "val_loss"
        )

        self.eval_checkpoint_value.config(
            text=format_tokens(
                tokens_seen
            )
        )

        self.eval_stored_val_value.config(
            text=(
                f"{float(stored_val):.4f}"
                if stored_val is not None
                else "-"
            )
        )

    def run_v01_generation(self):

        if (
            self.process is not None
            and self.process.poll() is None
        ):
            messagebox.showwarning(
                "Amber 0.1",
                "Une tâche Amber est déjà en cours."
            )
            return

        if not V01_CHECKPOINT.exists():
            messagebox.showwarning(
                "Amber 0.1",
                "Checkpoint Amber 0.1 introuvable."
            )
            return

        prompt = self.v01_prompt_input.get(
            "1.0",
            "end"
        ).strip()

        if not prompt:
            return

        try:
            temperature = float(
                self.v01_gen_temperature.get()
            )

            top_k = int(
                self.v01_gen_top_k.get()
            )

            top_p = float(
                self.v01_gen_top_p.get()
            )

            max_tokens = int(
                self.v01_gen_max_tokens.get()
            )

            repetition = float(
                self.v01_gen_repetition.get()
            )

        except ValueError:
            messagebox.showerror(
                "Amber 0.1",
                "Paramètres de génération invalides."
            )
            return

        mode = (
            "qa"
            if self.v01_gen_mode.get().startswith(
                "Question"
            )
            else "continuation"
        )

        self.v01_generation_output.delete(
            "1.0",
            "end"
        )

        self.v01_generation_output.insert(
            "end",
            "Chargement d'Amber 0.1..."
        )

        self.v01_eval_status.config(
            text="Génération en cours..."
        )

        self._run_command(
            [
                sys.executable,
                "-m",
                "inference.v01",
                "--prompt",
                prompt,
                "--mode",
                mode,
                "--max-new-tokens",
                str(
                    max(
                        1,
                        max_tokens
                    )
                ),
                "--temperature",
                str(
                    max(
                        0.0,
                        temperature
                    )
                ),
                "--top-k",
                str(
                    max(
                        0,
                        top_k
                    )
                ),
                "--top-p",
                str(
                    min(
                        max(
                            top_p,
                            0.0
                        ),
                        1.0
                    )
                ),
                "--repetition-penalty",
                str(
                    max(
                        1.0,
                        repetition
                    )
                ),
            ]
        )

    def run_v01_evaluation(self):

        if (
            self.process is not None
            and self.process.poll() is None
        ):
            messagebox.showwarning(
                "Amber 0.1",
                "Une tâche Amber est déjà en cours."
            )
            return

        if not V01_CHECKPOINT.exists():
            messagebox.showwarning(
                "Amber 0.1",
                "Checkpoint Amber 0.1 introuvable."
            )
            return

        try:
            batches = int(
                self.v01_eval_batches.get()
            )

        except ValueError:
            messagebox.showerror(
                "Amber 0.1",
                "Le nombre de batches doit être un entier."
            )
            return

        self.v01_eval_status.config(
            text="Évaluation validation en cours..."
        )

        self._run_command(
            [
                sys.executable,
                "-m",
                "evaluation.eval_v01",
                "--batches",
                str(
                    max(
                        1,
                        batches
                    )
                ),
                "--sequence-length",
                "512",
                "--batch-size",
                "4",
            ]
        )

    # ========================================================
    # TRAINING
    # ========================================================

    def _build_training(self):

        header = ttk.Frame(
            self.training_tab
        )

        header.pack(
            fill="x",
            pady=(20, 10)
        )

        ttk.Label(
            header,
            text="Amber Training",
            style="Title.TLabel"
        ).pack(
            anchor="w"
        )

        ttk.Label(
            header,
            text=(
                "Suivi en direct du modèle, Governor et progression."
            ),
            style="Subtitle.TLabel"
        ).pack(
            anchor="w",
            pady=(2, 10)
        )

        metrics = ttk.Frame(
            self.training_tab
        )

        metrics.pack(
            fill="x",
            pady=(0, 15)
        )

        self.train_step_value = self._card(
            metrics,
            "STEP"
        )

        self.train_loss_value = self._card(
            metrics,
            "LOSS"
        )

        self.train_speed_value = self._card(
            metrics,
            "VITESSE"
        )

        self.train_eta_value = self._card(
            metrics,
            "ETA"
        )

        self.train_vram_value = self._card(
            metrics,
            "VRAM TRAIN"
        )

        self.train_governor_value = self._card(
            metrics,
            "GOVERNOR"
        )

        governor_box = ttk.Frame(
            self.training_tab
        )

        governor_box.pack(
            fill="x",
            pady=5
        )

        ttk.Label(
            governor_box,
            text="Training Governor"
        ).pack(
            side="left",
            padx=(0, 20)
        )

        for mode in [
            "ECO",
            "BALANCED",
            "FULL"
        ]:
            ttk.Radiobutton(
                governor_box,
                text=mode,
                variable=self.mode,
                value=mode,
                command=lambda: self.train_governor_value.config(
                    text=self.mode.get()
                )
            ).pack(
                side="left",
                padx=(0, 20)
            )

        objective = ttk.Frame(
            self.training_tab
        )

        objective.pack(
            fill="x",
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
            objective,
            text="Prêt"
        )

        self.training_status.pack(
            side="left",
            padx=20
        )

        self.progress = ttk.Progressbar(
            self.training_tab,
            orient="horizontal",
            mode="determinate"
        )

        self.progress.pack(
            fill="x",
            pady=(10, 5)
        )

        progress_row = ttk.Frame(
            self.training_tab
        )

        progress_row.pack(
            fill="x"
        )

        self.progress_text = ttk.Label(
            progress_row,
            text="0 / 500"
        )

        self.progress_text.pack(
            side="left"
        )

        self.progress_percent = ttk.Label(
            progress_row,
            text="0.0 %"
        )

        self.progress_percent.pack(
            side="right"
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

        self.train_step_value.config(
            text=str(self.current_step)
        )

        self.train_loss_value.config(
            text="-"
        )

        self.train_speed_value.config(
            text="0.00 step/s"
        )

        self.train_eta_value.config(
            text="--:--"
        )

        self.train_vram_value.config(
            text="0.00 GB"
        )

        self.train_governor_value.config(
            text=self.mode.get()
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
            self.training_tab
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
    # DATASET MANAGER
    # ========================================================

    def _build_dataset(self):

        self.dataset_url = tk.StringVar(
            value=""
        )

        self.dataset_target_tokens = tk.StringVar(
            value="100000000"
        )

        self.tokenizer_vocab_size = tk.StringVar(
            value="32000"
        )

        self.dataset_last_plan = None
        self.dataset_catalog_keys = list(
            SOURCE_CATALOG.keys()
        )

        ttk.Label(
            self.dataset_tab,
            text="Dataset Planner 0.0.7",
            style="Title.TLabel"
        ).pack(
            anchor="w",
            pady=(16, 3)
        )

        ttk.Label(
            self.dataset_tab,
            text=(
                "Planifie le corpus Amber 0.1 avant de télécharger : "
                "taille, espace disque, licences et objectif de tokens."
            ),
            style="Subtitle.TLabel"
        ).pack(
            anchor="w",
            pady=(0, 10)
        )

        cards = ttk.Frame(
            self.dataset_tab
        )

        cards.pack(
            fill="x",
            pady=(0, 10)
        )

        self.dataset_sources_value = self._card(
            cards,
            "SOURCES LOCALES"
        )

        self.dataset_train_value = self._card(
            cards,
            "TRAIN"
        )

        self.dataset_validation_value = self._card(
            cards,
            "VALIDATION"
        )

        self.dataset_tokenizer_value = self._card(
            cards,
            "TOKENIZER"
        )

        self.dataset_target_value = self._card(
            cards,
            "CIBLE"
        )

        planner = ttk.Frame(
            self.dataset_tab
        )

        planner.pack(
            fill="x",
            pady=4
        )

        left = ttk.Frame(
            planner
        )

        left.pack(
            side="left",
            fill="y",
            padx=(0, 10)
        )

        ttk.Label(
            left,
            text="Corpus officiels sélectionnables"
        ).pack(
            anchor="w",
            pady=(0, 4)
        )

        self.dataset_catalog_list = tk.Listbox(
            left,
            selectmode="extended",
            exportselection=False,
            bg="#0b0b0f",
            fg="#ffffff",
            selectbackground="#34343e",
            selectforeground="#ffb347",
            relief="flat",
            font=("Segoe UI", 9),
            width=48,
            height=6
        )

        self.dataset_catalog_list.pack(
            fill="both",
            expand=True
        )

        for key in self.dataset_catalog_keys:
            source = SOURCE_CATALOG[key]

            self.dataset_catalog_list.insert(
                "end",
                source["name"]
            )

        target_box = ttk.Frame(
            left
        )

        target_box.pack(
            fill="x",
            pady=(6, 0)
        )

        ttk.Label(
            target_box,
            text="Objectif tokens :"
        ).pack(
            side="left"
        )

        ttk.Entry(
            target_box,
            textvariable=self.dataset_target_tokens,
            width=14
        ).pack(
            side="left",
            padx=6
        )

        ttk.Button(
            target_box,
            text="Estimer",
            command=self.estimate_dataset_plan
        ).pack(
            side="left",
            padx=4
        )

        right = ttk.Frame(
            planner
        )

        right.pack(
            side="left",
            fill="both",
            expand=True
        )

        ttk.Label(
            right,
            text="Plan / licences"
        ).pack(
            anchor="w",
            pady=(0, 4)
        )

        self.dataset_plan_text = ScrolledText(
            right,
            bg="#0b0b0f",
            fg="#e5e5ea",
            insertbackground="white",
            relief="flat",
            font=("Consolas", 9),
            height=8
        )

        self.dataset_plan_text.pack(
            fill="both",
            expand=True
        )

        action_row = ttk.Frame(
            self.dataset_tab
        )

        action_row.pack(
            fill="x",
            pady=(8, 4)
        )

        ttk.Button(
            action_row,
            text="Importer fichiers",
            command=self.import_dataset_files
        ).pack(
            side="left",
            padx=(0, 6)
        )

        ttk.Button(
            action_row,
            text="Télécharger sélection",
            command=self.download_selected_datasets
        ).pack(
            side="left",
            padx=6
        )

        ttk.Button(
            action_row,
            text="Annuler tâche",
            command=self.cancel_dataset_task
        ).pack(
            side="left",
            padx=6
        )

        ttk.Button(
            action_row,
            text="Préparer dataset",
            command=self.prepare_dataset_ui
        ).pack(
            side="left",
            padx=6
        )

        ttk.Button(
            action_row,
            text="Préparer Amber 0.1",
            command=self.prepare_amber_01
        ).pack(
            side="left",
            padx=6
        )

        custom = ttk.Frame(
            self.dataset_tab
        )

        custom.pack(
            fill="x",
            pady=4
        )

        ttk.Entry(
            custom,
            textvariable=self.dataset_url
        ).pack(
            side="left",
            fill="x",
            expand=True,
            padx=(0, 6)
        )

        ttk.Button(
            custom,
            text="Télécharger URL",
            command=self.download_custom_dataset
        ).pack(
            side="left",
            padx=4
        )

        ttk.Label(
            custom,
            text="Vocab :"
        ).pack(
            side="left",
            padx=(12, 4)
        )

        ttk.Entry(
            custom,
            textvariable=self.tokenizer_vocab_size,
            width=9
        ).pack(
            side="left",
            padx=(0, 4)
        )

        ttk.Button(
            custom,
            text="Entraîner tokenizer",
            command=self.train_tokenizer_ui
        ).pack(
            side="left",
            padx=4
        )

        self.dataset_progress_label = ttk.Label(
            self.dataset_tab,
            text="Prêt"
        )

        self.dataset_progress_label.pack(
            anchor="w",
            pady=(6, 2)
        )

        self.dataset_status = ScrolledText(
            self.dataset_tab,
            bg="#0b0b0f",
            fg="#e5e5ea",
            insertbackground="white",
            relief="flat",
            font=("Consolas", 9),
            height=9
        )

        self.dataset_status.pack(
            fill="both",
            expand=True,
            pady=(2, 0)
        )

        self.refresh_dataset()

    def _dataset_selected_keys(self):

        return [
            self.dataset_catalog_keys[index]
            for index in self.dataset_catalog_list.curselection()
        ]

    def _dataset_target_tokens_value(self):

        value = self.dataset_target_tokens.get().strip()

        try:
            tokens = int(
                value
            )

        except ValueError:
            raise ValueError(
                "L'objectif tokens doit être un nombre entier."
            )

        if tokens < 1_000_000:
            raise ValueError(
                "Utilise au moins 1 000 000 tokens pour le Planner."
            )

        return tokens

    def _dataset_log(
        self,
        text
    ):
        self.dataset_status.insert(
            "end",
            self._timestamped_line(
                str(text)
            )
        )

        self.dataset_status.see(
            "end"
        )

    def _dataset_plan_log(
        self,
        text
    ):
        self.dataset_plan_text.insert(
            "end",
            self._timestamped_line(
                str(text)
            )
        )

        self.dataset_plan_text.see(
            "end"
        )

    def _dataset_progress(
        self,
        text
    ):
        self.dataset_progress_label.config(
            text=str(text)
        )

    def _run_dataset_task(
        self,
        label,
        worker,
        done_message=None
    ):
        if self.dataset_busy:
            messagebox.showwarning(
                "Amber Dataset Manager",
                "Une tâche Dataset est déjà en cours."
            )
            return

        self.dataset_busy = True
        self.dataset_cancel_event.clear()

        self._dataset_progress(
            label
        )

        def thread_worker():
            try:
                result = worker()

                def finish():
                    self.dataset_busy = False

                    if done_message:
                        message = (
                            done_message(result)
                            if callable(done_message)
                            else str(done_message)
                        )

                        self._dataset_log(
                            message
                        )

                    self._dataset_progress(
                        "Prêt"
                    )

                    self.refresh_dataset()

                self.after(
                    0,
                    finish
                )

            except DownloadCancelled as exc:

                def cancelled():
                    self.dataset_busy = False

                    self._dataset_progress(
                        "Annulé — reprise possible"
                    )

                    self._dataset_log(
                        str(exc)
                    )

                    self.refresh_dataset()

                self.after(
                    0,
                    cancelled
                )

            except Exception as exc:

                def fail():
                    self.dataset_busy = False

                    self._dataset_progress(
                        "Erreur"
                    )

                    self._dataset_log(
                        f"[ERREUR] {exc}"
                    )

                    messagebox.showerror(
                        "Amber Dataset Manager",
                        str(exc)
                    )

                self.after(
                    0,
                    fail
                )

        threading.Thread(
            target=thread_worker,
            daemon=True
        ).start()

    def cancel_dataset_task(self):

        if not self.dataset_busy:
            self._dataset_progress(
                "Aucune tâche à annuler"
            )
            return

        self.dataset_cancel_event.set()

        self._dataset_progress(
            "Annulation demandée..."
        )

    def import_dataset_files(self):

        files = filedialog.askopenfilenames(
            title="Importer des données dans Amber",
            filetypes=[
                (
                    "Textes / datasets",
                    "*.txt *.md *.jsonl *.xml *.gz *.bz2"
                ),
                (
                    "Tous les fichiers",
                    "*.*"
                ),
            ]
        )

        if not files:
            return

        def worker():
            imported = []

            for path in files:
                if self.dataset_cancel_event.is_set():
                    raise DownloadCancelled(
                        "Import annulé."
                    )

                imported.append(
                    import_local_file(
                        path
                    )
                )

            return imported

        self._run_dataset_task(
            "Import en cours...",
            worker,
            lambda result: (
                f"{len(result)} source(s) importée(s)."
            )
        )

    def estimate_dataset_plan(self):

        if self.dataset_busy:
            messagebox.showwarning(
                "Amber Dataset Planner",
                "Une tâche Dataset est déjà en cours."
            )
            return

        keys = self._dataset_selected_keys()

        if not keys:
            messagebox.showwarning(
                "Amber Dataset Planner",
                "Sélectionne au moins un corpus."
            )
            return

        try:
            target_tokens = self._dataset_target_tokens_value()

        except ValueError as exc:
            messagebox.showerror(
                "Amber Dataset Planner",
                str(exc)
            )
            return

        self.dataset_busy = True
        self.dataset_cancel_event.clear()
        self._dataset_progress(
            "Estimation distante..."
        )

        def worker():
            try:
                plan = estimate_source_plan(
                    keys,
                    target_tokens=target_tokens
                )

                def finish():
                    self.dataset_busy = False
                    self.dataset_last_plan = plan

                    self.dataset_plan_text.delete(
                        "1.0",
                        "end"
                    )

                    self._dataset_plan_log(
                        (
                            f"Cible : {plan['target_tokens_human']} tokens "
                            f"(~{plan['estimated_text_human']} de texte)"
                        )
                    )

                    self._dataset_plan_log(
                        (
                            f"Téléchargements connus restants : "
                            f"{plan['known_remaining_human']}"
                        )
                    )

                    if plan["unknown_sizes"]:
                        self._dataset_plan_log(
                            (
                                f"Tailles distantes inconnues : "
                                f"{plan['unknown_sizes']}"
                            )
                        )

                    self._dataset_plan_log(
                        (
                            f"Espace libre recommandé : "
                            f"{plan['recommended_free_human']}"
                        )
                    )

                    self._dataset_plan_log(
                        (
                            f"Espace libre actuel : "
                            f"{plan['disk_free_human']} "
                            f"({'OK' if plan['disk_ok'] else 'INSUFFISANT'})"
                        )
                    )

                    self._dataset_plan_log(
                        ""
                    )

                    for item in plan["items"]:
                        self._dataset_plan_log(
                            (
                                f"- {item['name']} : "
                                f"{item['size_human']} "
                                f"(reste {item['remaining_human']})"
                            )
                        )

                        self._dataset_plan_log(
                            f"  Licence : {item['license']}"
                        )

                        if item.get("note"):
                            self._dataset_plan_log(
                                f"  Note : {item['note']}"
                            )

                    write_manifest(
                        planner=plan
                    )

                    self._dataset_progress(
                        "Plan calculé"
                    )

                    self._dataset_log(
                        (
                            "Plan calculé : espace recommandé "
                            f"{plan['recommended_free_human']}."
                        )
                    )

                    self.refresh_dataset()

                self.after(
                    0,
                    finish
                )

            except Exception as exc:

                def fail():
                    self.dataset_busy = False
                    self._dataset_progress(
                        "Erreur d'estimation"
                    )
                    self._dataset_log(
                        f"[ERREUR] {exc}"
                    )
                    messagebox.showerror(
                        "Amber Dataset Planner",
                        str(exc)
                    )

                self.after(
                    0,
                    fail
                )

        threading.Thread(
            target=worker,
            daemon=True
        ).start()

    def download_selected_datasets(self):

        keys = self._dataset_selected_keys()

        if not keys:
            messagebox.showwarning(
                "Amber Dataset Manager",
                "Sélectionne au moins un corpus."
            )
            return

        source_lines = []

        for key in keys:
            source = SOURCE_CATALOG[key]

            source_lines.append(
                (
                    f"• {source['name']}\n"
                    f"  {source['license']}"
                )
            )

        if not messagebox.askyesno(
            "Amber Dataset Manager",
            (
                "Télécharger les corpus sélectionnés ?\n\n"
                + "\n".join(source_lines)
                + "\n\nLes téléchargements interrompus "
                "peuvent reprendre grâce aux fichiers .part."
            )
        ):
            return

        def progress(
            info
        ):
            downloaded = info.get(
                "downloaded",
                0
            )

            total = info.get(
                "total"
            )

            if total:
                percent = (
                    downloaded / total
                ) * 100

                text = (
                    f"{info['index']}/{info['count']} "
                    f"{info['name']} : "
                    f"{format_bytes(downloaded)} / "
                    f"{format_bytes(total)} "
                    f"({percent:.1f} %)"
                )

            else:
                text = (
                    f"{info['index']}/{info['count']} "
                    f"{info['name']} : "
                    f"{format_bytes(downloaded)}"
                )

            self.after(
                0,
                lambda value=text: self._dataset_progress(
                    value
                )
            )

        self._run_dataset_task(
            "Téléchargement...",
            lambda: download_catalog_sources(
                keys,
                progress_callback=progress,
                cancel_event=self.dataset_cancel_event
            ),
            lambda result: (
                f"{len(result)} corpus téléchargé(s)/déjà présent(s)."
            )
        )

    def download_custom_dataset(self):

        url = self.dataset_url.get().strip()

        if not url:
            messagebox.showwarning(
                "Amber Dataset Manager",
                "Colle d'abord une URL HTTP/HTTPS."
            )
            return

        if not messagebox.askyesno(
            "Amber Dataset Manager",
            (
                "Télécharger cette source ?\n\n"
                "Vérifie toi-même que la licence autorise "
                "l'utilisation prévue."
            )
        ):
            return

        def progress(
            downloaded,
            total
        ):
            if total:
                percent = (
                    downloaded / total
                ) * 100

                text = (
                    f"Téléchargement : "
                    f"{format_bytes(downloaded)} / "
                    f"{format_bytes(total)} "
                    f"({percent:.1f} %)"
                )

            else:
                text = (
                    "Téléchargement : "
                    f"{format_bytes(downloaded)}"
                )

            self.after(
                0,
                lambda value=text: self._dataset_progress(
                    value
                )
            )

        self._run_dataset_task(
            "Téléchargement URL...",
            lambda: download_source(
                url,
                progress_callback=progress,
                cancel_event=self.dataset_cancel_event,
                resume=True
            ),
            lambda result: (
                f"Téléchargé : {result['name']} "
                f"({result['size_human']})"
            )
        )

    def prepare_dataset_ui(self):

        try:
            target_tokens = self._dataset_target_tokens_value()

        except ValueError as exc:
            messagebox.showerror(
                "Amber Dataset Manager",
                str(exc)
            )
            return

        if not messagebox.askyesno(
            "Amber Dataset Manager",
            (
                "Préparer le dataset ?\n\n"
                "Amber va nettoyer les textes, dédupliquer, "
                "créer le split train/validation 98/2 et viser "
                f"environ {format_tokens(target_tokens)} tokens."
            )
        ):
            return

        def progress(
            info
        ):
            approx_tokens = info.get(
                "approx_tokens",
                0
            )

            text = (
                f"Préparation : "
                f"{format_tokens(approx_tokens)} tokens approx. | "
                f"{info.get('train_docs', 0):,} train | "
                f"{info.get('validation_docs', 0):,} validation"
            )

            self.after(
                0,
                lambda value=text: self._dataset_progress(
                    value
                )
            )

        self._run_dataset_task(
            "Préparation dataset...",
            lambda: prepare_dataset(
                target_tokens=target_tokens,
                progress_callback=progress,
                cancel_event=self.dataset_cancel_event
            ),
            lambda result: (
                "Dataset prêt : "
                f"~{result['approx_tokens_human']} tokens, "
                f"{result['train_docs']:,} docs train, "
                f"{result['validation_docs']:,} validation."
            )
        )

    def train_tokenizer_ui(self):

        try:
            vocab_size = int(
                self.tokenizer_vocab_size.get()
            )

        except ValueError:
            messagebox.showerror(
                "Amber Tokenizer",
                "Le vocabulaire doit être un nombre entier."
            )
            return

        if not PREPARED_TRAIN_FILE.exists():
            messagebox.showwarning(
                "Amber Tokenizer",
                "Prépare d'abord le dataset."
            )
            return

        if not messagebox.askyesno(
            "Amber Tokenizer",
            (
                f"Entraîner AmberBPETokenizer avec "
                f"un vocabulaire cible de {vocab_size:,} tokens ?\n\n"
                "Aucun tokenizer pré-entraîné ni poids d'un autre LLM "
                "ne seront utilisés."
            )
        ):
            return

        def progress(
            message
        ):
            self.after(
                0,
                lambda value=message: self._dataset_progress(
                    value
                )
            )

        self._run_dataset_task(
            "Entraînement tokenizer...",
            lambda: train_amber_tokenizer(
                vocab_size=vocab_size,
                progress_callback=progress
            ),
            lambda result: (
                "Tokenizer Amber entraîné : "
                f"{result['actual_vocab_size']:,} tokens."
            )
        )

    def prepare_amber_01(self):

        sources = list_raw_sources()

        if not sources:
            messagebox.showwarning(
                "Préparer Amber 0.1",
                (
                    "Ajoute d'abord au moins une vraie source. "
                    "Le seed de 3,9 KB ne suffit pas pour Amber 0.1."
                )
            )
            return

        try:
            target_tokens = self._dataset_target_tokens_value()

        except ValueError as exc:
            messagebox.showerror(
                "Préparer Amber 0.1",
                str(exc)
            )
            return

        if not messagebox.askyesno(
            "Préparer Amber 0.1",
            (
                "Lancer le pipeline Amber 0.1 ?\n\n"
                f"Objectif : ~{format_tokens(target_tokens)} tokens\n"
                "1. Nettoyage + déduplication\n"
                "2. Split train/validation\n"
                "3. Tokenizer BPE 32k entraîné depuis zéro\n"
                "4. Manifest de préparation\n\n"
                "Cette étape peut être longue."
            )
        ):
            return

        self.tokenizer_vocab_size.set(
            "32000"
        )

        def worker():

            def prep_progress(
                info
            ):
                approx_tokens = info.get(
                    "approx_tokens",
                    0
                )

                self.after(
                    0,
                    lambda value=(
                        f"Amber 0.1 — dataset : "
                        f"{format_tokens(approx_tokens)} tokens approx."
                    ): self._dataset_progress(
                        value
                    )
                )

            prepared = prepare_dataset(
                target_tokens=target_tokens,
                progress_callback=prep_progress,
                cancel_event=self.dataset_cancel_event
            )

            if self.dataset_cancel_event.is_set():
                raise DownloadCancelled(
                    "Pipeline Amber 0.1 annulé."
                )

            self.after(
                0,
                lambda: self._dataset_progress(
                    "Amber 0.1 — entraînement tokenizer 32k..."
                )
            )

            tokenizer = train_amber_tokenizer(
                vocab_size=32000
            )

            manifest = write_manifest(
                prepared=prepared,
                planner=self.dataset_last_plan
            )

            return {
                "prepared": prepared,
                "tokenizer": tokenizer,
                "manifest": manifest,
            }

        self._run_dataset_task(
            "Préparation Amber 0.1...",
            worker,
            lambda result: (
                "Amber 0.1 prêt côté données : "
                f"~{result['prepared']['approx_tokens_human']} tokens, "
                f"tokenizer {result['tokenizer']['actual_vocab_size']:,}."
            )
        )

    def create_dataset_manifest(self):

        try:
            manifest = write_manifest(
                planner=self.dataset_last_plan
            )

            self._dataset_log(
                "Manifest créé : data/dataset_manifest.json"
            )

            self._dataset_log(
                f"Statut : {manifest['status']}"
            )

            self.refresh_dataset()

        except Exception as exc:
            messagebox.showerror(
                "Amber Dataset Manager",
                str(exc)
            )

    def refresh_dataset(self):

        sources = list_raw_sources()

        prepared_train = file_stats(
            PREPARED_TRAIN_FILE
        )

        prepared_validation = file_stats(
            PREPARED_VALIDATION_FILE
        )

        tok = tokenizer_status()

        total_source_size = sum(
            item["size_bytes"]
            for item in sources
        )

        self.dataset_sources_value.config(
            text=(
                f"{len(sources)} / "
                f"{format_bytes(total_source_size)}"
            )
        )

        self.dataset_train_value.config(
            text=(
                prepared_train["size_human"]
                if prepared_train["exists"]
                else "-"
            )
        )

        self.dataset_validation_value.config(
            text=(
                prepared_validation["size_human"]
                if prepared_validation["exists"]
                else "-"
            )
        )

        self.dataset_tokenizer_value.config(
            text=(
                f"{tok['vocab_size']:,}"
                if tok["exists"]
                else "Non entraîné"
            )
        )

        try:
            target_tokens = self._dataset_target_tokens_value()

            self.dataset_target_value.config(
                text=format_tokens(
                    target_tokens
                )
            )

        except Exception:
            self.dataset_target_value.config(
                text="-"
            )

        self.dataset_status.delete(
            "1.0",
            "end"
        )

        self._dataset_log(
            "AMBER DATASET PLANNER 0.0.7"
        )

        self._dataset_log(
            "=" * 56
        )

        self._dataset_log(
            (
                f"Sources locales : {len(sources)} "
                f"({format_bytes(total_source_size)})"
            )
        )

        self._dataset_log(
            (
                "Train préparé : "
                + (
                    prepared_train["size_human"]
                    if prepared_train["exists"]
                    else "absent"
                )
            )
        )

        self._dataset_log(
            (
                "Validation : "
                + (
                    prepared_validation["size_human"]
                    if prepared_validation["exists"]
                    else "absente"
                )
            )
        )

        if tok["exists"]:
            self._dataset_log(
                (
                    "Tokenizer Amber : prêt, "
                    f"{tok['vocab_size']:,} tokens"
                )
            )

        else:
            self._dataset_log(
                "Tokenizer Amber : non entraîné"
            )

        self._dataset_log(
            ""
        )

        self._dataset_log(
            "Les téléchargements .part sont repris automatiquement."
        )

        self._dataset_log(
            "L'objectif tokens reste une estimation avant encodage BPE exact."
        )

    # ========================================================
    # AMBER 0.1
    # ========================================================

    def _build_v01(self):

        ttk.Label(
            self.v01_tab,
            text="Amber 0.1.7",
            style="Title.TLabel"
        ).pack(
            anchor="w",
            pady=(18, 4)
        )

        ttk.Label(
            self.v01_tab,
            text=(
                "Premier modèle Amber ~100 M paramètres, "
                "initialisé depuis zéro avec RoPE, GQA, RMSNorm et SwiGLU."
            ),
            style="Subtitle.TLabel"
        ).pack(
            anchor="w",
            pady=(0, 12)
        )

        cards = ttk.Frame(
            self.v01_tab
        )

        cards.pack(
            fill="x",
            pady=(0, 12)
        )

        self.v01_model_value = self._card(
            cards,
            "PARAMÈTRES"
        )

        self.v01_vocab_value = self._card(
            cards,
            "VOCAB"
        )

        self.v01_context_value = self._card(
            cards,
            "CONTEXTE"
        )

        self.v01_cache_value = self._card(
            cards,
            "TOKENS DATASET"
        )

        self.v01_seen_value = self._card(
            cards,
            "TOKENS VUS"
        )

        self.v01_loss_value = self._card(
            cards,
            "LOSS / VAL"
        )

        controls = ttk.Frame(
            self.v01_tab
        )

        controls.pack(
            fill="x",
            pady=5
        )

        ttk.Button(
            controls,
            text="Construire cache tokens",
            command=self.build_v01_cache
        ).pack(
            side="left",
            padx=(0, 7)
        )

        ttk.Button(
            controls,
            text="Tester Amber 0.1",
            command=self.test_v01_model
        ).pack(
            side="left",
            padx=7
        )

        ttk.Button(
            controls,
            text="Auto-Tuner RX 7900 XT",
            command=self.run_v01_autotune
        ).pack(
            side="left",
            padx=7
        )

        ttk.Label(
            controls,
            text="Objectif arrêt :"
        ).pack(
            side="left",
            padx=(18, 4)
        )

        ttk.Entry(
            controls,
            textvariable=self.v01_target_tokens,
            width=13
        ).pack(
            side="left",
            padx=(0, 8)
        )

        ttk.Label(
            controls,
            text="Horizon LR :"
        ).pack(
            side="left",
            padx=(8, 4)
        )

        ttk.Entry(
            controls,
            textvariable=self.v01_schedule_tokens,
            width=13
        ).pack(
            side="left",
            padx=(0, 8)
        )

        mode_row = ttk.Frame(
            self.v01_tab
        )

        mode_row.pack(
            fill="x",
            pady=6
        )

        ttk.Label(
            mode_row,
            text="Profil GPU :"
        ).pack(
            side="left",
            padx=(0, 10)
        )

        ttk.Label(
            mode_row,
            text="Cible VRAM FULL :"
        ).pack(
            side="left",
            padx=(10, 4)
        )

        ttk.Combobox(
            mode_row,
            textvariable=self.v01_full_vram_target,
            values=[
                "AUTO",
                "10",
                "15"
            ],
            state="readonly",
            width=6
        ).pack(
            side="left",
            padx=(0, 10)
        )

        for mode in (
            "ECO",
            "BALANCED",
            "FULL"
        ):
            ttk.Radiobutton(
                mode_row,
                text=mode,
                variable=self.v01_mode,
                value=mode
            ).pack(
                side="left",
                padx=(0, 14)
            )

        ttk.Button(
            mode_row,
            text="Démarrer / Reprendre",
            command=self.start_v01_training
        ).pack(
            side="left",
            padx=(15, 7)
        )

        ttk.Button(
            mode_row,
            text="Nouveau run",
            command=lambda: self.start_v01_training(
                fresh=True
            )
        ).pack(
            side="left",
            padx=7
        )

        self.v01_pause_button = ttk.Button(
            mode_row,
            text="Pause",
            command=self.toggle_v01_pause
        )

        self.v01_pause_button.pack(
            side="left",
            padx=7
        )

        ttk.Button(
            mode_row,
            text="Sauvegarder et arrêter",
            command=self.stop_v01_training
        ).pack(
            side="left",
            padx=7
        )

        ttk.Button(
            mode_row,
            text="Forcer arrêt",
            command=self.force_stop_v01_training
        ).pack(
            side="left",
            padx=7
        )

        self.v01_status_label = ttk.Label(
            self.v01_tab,
            text="Prêt"
        )

        self.v01_status_label.pack(
            anchor="w",
            pady=(7, 3)
        )

        self.v01_progress = ttk.Progressbar(
            self.v01_tab,
            orient="horizontal",
            mode="determinate"
        )

        self.v01_progress.pack(
            fill="x",
            pady=(2, 3)
        )

        self.v01_progress_text = ttk.Label(
            self.v01_tab,
            text="0 / 100.0 M tokens"
        )

        self.v01_progress_text.pack(
            anchor="w"
        )

        self.v01_schedule_text = ttk.Label(
            self.v01_tab,
            text="Horizon LR : 100.0 M tokens"
        )

        self.v01_schedule_text.pack(
            anchor="w",
            pady=(1, 0)
        )

        self.v01_autotune_text = ttk.Label(
            self.v01_tab,
            text="Auto-Tuner FULL : non lancé"
        )

        self.v01_autotune_text.pack(
            anchor="w",
            pady=(1, 0)
        )

        self.v01_vram_target_text = ttk.Label(
            self.v01_tab,
            text="FULL : cible VRAM 15 GB"
        )

        self.v01_vram_target_text.pack(
            anchor="w",
            pady=(1, 0)
        )

        self.v01_metrics = ttk.Label(
            self.v01_tab,
            text="Vitesse : - | VRAM : - | ETA : -"
        )

        self.v01_metrics.pack(
            anchor="w",
            pady=(2, 7)
        )

        self.v01_console = ScrolledText(
            self.v01_tab,
            bg="#0b0b0f",
            fg="#e5e5ea",
            insertbackground="white",
            relief="flat",
            font=("Consolas", 9),
            height=16
        )

        self.v01_console.pack(
            fill="both",
            expand=True
        )

        self.refresh_v01_status()

    def _v01_log(
        self,
        text
    ):
        self.v01_console.insert(
            "end",
            self._timestamped_line(
                str(text)
            )
        )

        self.v01_console.see(
            "end"
        )

    def _read_v01_status_file(self):

        if not V01_STATUS_FILE.exists():
            return {}

        try:
            return json.loads(
                V01_STATUS_FILE.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            return {}

    def _read_v01_autotune_file(self):

        if not V01_AUTOTUNE_FILE.exists():
            return {}

        try:
            return json.loads(
                V01_AUTOTUNE_FILE.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            return {}

    def refresh_v01_status(self):

        tok = tokenizer_status()

        vocab_size = (
            int(tok["vocab_size"])
            if tok.get("exists")
            and tok.get("vocab_size")
            else 32000
        )

        config = AmberV01Config(
            vocab_size=vocab_size
        )

        params = estimate_v01_parameter_count(
            config
        )

        cache = load_v01_cache_metadata()
        status = self._read_v01_status_file()
        autotune = self._read_v01_autotune_file()

        full_vram_target = self.v01_full_vram_target.get().strip()

        self.v01_vram_target_text.config(
            text=(
                "FULL : "
                + (
                    "priorité vitesse Auto-Tuner"
                    if full_vram_target.upper() == "AUTO"
                    else f"cible VRAM {full_vram_target} GB"
                )
            )
        )

        if autotune.get("best"):
            best = autotune["best"]

            self.v01_autotune_text.config(
                text=(
                    "Auto-Tuner FULL : "
                    f"MB {int(best.get('micro_batch', 1))} × "
                    f"accum {int(best.get('gradient_accumulation', 16))} | "
                    f"checkpoint "
                    f"{'ON' if best.get('checkpointing', True) else 'OFF'} | "
                    f"{float(best.get('tokens_per_second', 0)):,.0f} tok/s"
                )
            )

        else:
            self.v01_autotune_text.config(
                text="Auto-Tuner FULL : non lancé"
            )

        self.v01_model_value.config(
            text=f"{params / 1_000_000:.2f} M"
        )

        self.v01_vocab_value.config(
            text=f"{vocab_size:,}"
        )

        self.v01_context_value.config(
            text=str(
                config.context_length
            )
        )

        train_tokens = 0

        if cache:
            train_tokens = int(
                cache.get(
                    "train",
                    {}
                ).get(
                    "tokens",
                    0
                )
            )

        self.v01_cache_value.config(
            text=(
                format_tokens(train_tokens)
                if train_tokens
                else "À construire"
            )
        )

        tokens_seen = int(
            status.get(
                "tokens_seen",
                0
            )
        )

        self.v01_seen_value.config(
            text=format_tokens(
                tokens_seen
            )
        )

        train_loss = status.get(
            "train_loss"
        )

        val_loss = status.get(
            "val_loss"
        )

        if train_loss is None:
            loss_text = "-"
        else:
            loss_text = (
                f"{float(train_loss):.3f} / "
                + (
                    f"{float(val_loss):.3f}"
                    if val_loss is not None
                    else "-"
                )
            )

        self.v01_loss_value.config(
            text=loss_text
        )

        try:
            target = int(
                self.v01_target_tokens.get()
            )
        except Exception:
            target = 100_000_000

        try:
            requested_schedule = int(
                self.v01_schedule_tokens.get()
            )
        except Exception:
            requested_schedule = 100_000_000

        saved_schedule = status.get(
            "schedule_tokens"
        )

        schedule_display = (
            int(saved_schedule)
            if saved_schedule
            else requested_schedule
        )

        self.v01_schedule_text.config(
            text=(
                "Horizon LR : "
                f"{format_tokens(schedule_display)} tokens"
            )
        )

        self.v01_progress.configure(
            maximum=max(
                target,
                1
            ),
            value=min(
                tokens_seen,
                target
            )
        )

        self.v01_progress_text.config(
            text=(
                f"{format_tokens(tokens_seen)} / "
                f"{format_tokens(target)} tokens"
            )
        )

        if cache:
            self.v01_status_label.config(
                text=(
                    "Cache prêt — "
                    f"{cache.get('train', {}).get('tokens', 0):,} "
                    "tokens train exacts"
                )
            )
        else:
            self.v01_status_label.config(
                text=(
                    "Construis d'abord le cache tokens BPE exact."
                )
            )

    def build_v01_cache(self):

        if (
            self.process is not None
            and self.process.poll() is None
        ):
            messagebox.showwarning(
                "Amber 0.1",
                "Une tâche Amber est déjà en cours."
            )
            return

        tok = tokenizer_status()

        if (
            not tok.get("exists")
            or int(
                tok.get(
                    "vocab_size",
                    0
                )
            ) < 1000
        ):
            messagebox.showwarning(
                "Amber 0.1",
                (
                    "Le tokenizer 32k Amber n'est pas prêt. "
                    "Utilise d'abord Préparer Amber 0.1 dans Dataset."
                )
            )
            return

        self.v01_console.delete(
            "1.0",
            "end"
        )

        self.v01_status_label.config(
            text="Construction du cache tokens..."
        )

        self._run_command(
            [
                sys.executable,
                "-m",
                "training.data_v01",
                "--build"
            ]
        )

    def test_v01_model(self):

        tok = tokenizer_status()

        if not tok.get("exists"):
            messagebox.showwarning(
                "Amber 0.1",
                "Tokenizer Amber absent."
            )
            return

        self.v01_status_label.config(
            text="Smoke test GPU..."
        )

        self._run_command(
            [
                sys.executable,
                "-m",
                "tests.test_v01"
            ]
        )

    def run_v01_autotune(self):

        if (
            self.process is not None
            and self.process.poll() is None
        ):
            messagebox.showwarning(
                "Amber Auto-Tuner",
                (
                    "Arrête proprement l'entraînement avant de lancer "
                    "l'Auto-Tuner."
                )
            )
            return

        if not V01_CHECKPOINT.exists():
            messagebox.showwarning(
                "Amber Auto-Tuner",
                (
                    "Aucun checkpoint Amber 0.1 actif. "
                    "Lance d'abord un entraînement et sauvegarde-le."
                )
            )
            return

        if not messagebox.askyesno(
            "Amber Auto-Tuner RX 7900 XT",
            (
                "Tester automatiquement plusieurs configurations FULL ?\n\n"
                "Amber va comparer micro-batch 1/2/4/8/16, avec et sans "
                "gradient checkpointing, sans modifier ni sauvegarder "
                "les poids du checkpoint.\n\n"
                "Une marge de sécurité de 10 % de VRAM est conservée."
            )
        ):
            return

        self.v01_status_label.config(
            text="Auto-Tuner RX 7900 XT en cours..."
        )

        self._v01_log(
            "[AUTOTUNE] Démarrage du benchmark FULL."
        )

        self._run_command(
            [
                sys.executable,
                "-m",
                "training.autotune_v01"
            ]
        )

    def start_v01_training(
        self,
        fresh=False
    ):

        if (
            self.process is not None
            and self.process.poll() is None
        ):
            messagebox.showwarning(
                "Amber 0.1",
                "Une tâche Amber est déjà en cours."
            )
            return

        cache = load_v01_cache_metadata()

        if not cache:
            messagebox.showwarning(
                "Amber 0.1",
                "Construis d'abord le cache tokens."
            )
            return

        try:
            target = int(
                self.v01_target_tokens.get()
            )

            schedule = int(
                self.v01_schedule_tokens.get()
            )

        except ValueError:
            messagebox.showerror(
                "Amber 0.1",
                "Objectif arrêt et Horizon LR doivent être des entiers."
            )
            return

        if target <= 0 or schedule <= 0:
            messagebox.showerror(
                "Amber 0.1",
                "Les valeurs doivent être supérieures à zéro."
            )
            return

        if target > schedule:
            messagebox.showerror(
                "Amber 0.1",
                (
                    "L'objectif d'arrêt ne peut pas dépasser "
                    "l'Horizon LR."
                )
            )
            return

        status = self._read_v01_status_file()
        existing_tokens = int(
            status.get(
                "tokens_seen",
                0
            )
        )

        existing_schedule = status.get(
            "schedule_tokens"
        )

        if fresh and existing_tokens > 0:
            if not messagebox.askyesno(
                "Nouveau run Amber 0.1",
                (
                    f"Archiver le checkpoint actuel "
                    f"({format_tokens(existing_tokens)} tokens) "
                    "et redémarrer depuis zéro ?\n\n"
                    "Le checkpoint benchmark sera conservé dans "
                    "le dossier checkpoints."
                )
            ):
                return

        if (
            not fresh
            and existing_tokens > 0
            and existing_schedule is None
        ):
            messagebox.showwarning(
                "Amber 0.1",
                (
                    "Le checkpoint actuel provient du benchmark 0.1.3 "
                    "et n'a pas d'Horizon LR fixe.\n\n"
                    "Utilise 'Nouveau run' pour l'archiver et lancer "
                    "le vrai entraînement avec un scheduler propre."
                )
            )
            return

        if (
            not fresh
            and existing_tokens > 0
            and existing_schedule is not None
            and int(existing_schedule) != schedule
        ):
            messagebox.showwarning(
                "Amber 0.1",
                (
                    "L'Horizon LR ne correspond pas au checkpoint.\n\n"
                    f"Checkpoint : {format_tokens(int(existing_schedule))}\n"
                    f"Demandé : {format_tokens(schedule)}\n\n"
                    "Garde le même horizon pour reprendre, ou utilise "
                    "'Nouveau run'."
                )
            )
            return

        for control_file in (
            V01_STOP_FILE,
            V01_PAUSE_FILE
        ):
            try:
                if control_file.exists():
                    control_file.unlink()
            except Exception:
                pass

        self.v01_paused = False

        self.v01_pause_button.config(
            text="Pause"
        )

        self.v01_status_label.config(
            text="Pré-entraînement en cours..."
        )

        self.v01_schedule_text.config(
            text=(
                "Horizon LR : "
                f"{format_tokens(schedule)} tokens"
            )
        )

        command = [
            sys.executable,
            "-m",
            "training.train_v01",
            "--mode",
            self.v01_mode.get().lower(),
            "--target-tokens",
            str(target),
            "--schedule-tokens",
            str(schedule)
        ]

        if self.v01_mode.get().upper() == "FULL":
            full_vram_target = (
                self.v01_full_vram_target.get().strip()
            )

            if full_vram_target.upper() != "AUTO":
                try:
                    target_gb = float(
                        full_vram_target
                    )
                except ValueError:
                    target_gb = 15.0

                command.extend(
                    [
                        "--full-vram-target",
                        str(target_gb)
                    ]
                )

                self._v01_log(
                    (
                        "[V01] FULL cible VRAM : "
                        f"{target_gb:.1f} GB."
                    )
                )

        if fresh or existing_tokens <= 0:
            command.append(
                "--fresh"
            )

            if fresh:
                self._v01_log(
                    (
                        "[V01] Nouveau run demandé : "
                        f"objectif={format_tokens(target)}, "
                        f"horizon LR={format_tokens(schedule)}."
                    )
                )

            else:
                self._v01_log(
                    "[V01] Aucun apprentissage validé : démarrage propre demandé."
                )

        else:
            self._v01_log(
                (
                    "[V01] Reprise du checkpoint : "
                    f"{format_tokens(existing_tokens)} tokens, "
                    f"horizon LR={format_tokens(schedule)}."
                )
            )

        self._run_command(
            command
        )

    def toggle_v01_pause(self):

        if (
            self.process is None
            or self.process.poll() is not None
        ):
            return

        try:
            if not self.v01_paused:
                V01_PAUSE_FILE.write_text(
                    "pause",
                    encoding="utf-8"
                )

                self.v01_paused = True

                self.v01_pause_button.config(
                    text="Reprendre"
                )

                self.v01_status_label.config(
                    text="Pause demandée..."
                )

            else:
                if V01_PAUSE_FILE.exists():
                    V01_PAUSE_FILE.unlink()

                self.v01_paused = False

                self.v01_pause_button.config(
                    text="Pause"
                )

                self.v01_status_label.config(
                    text="Reprise..."
                )

        except Exception as exc:
            messagebox.showerror(
                "Amber 0.1",
                str(exc)
            )

    def stop_v01_training(self):

        if (
            self.process is None
            or self.process.poll() is not None
        ):
            return

        try:
            V01_STOP_FILE.write_text(
                "stop",
                encoding="utf-8"
            )

            self.v01_status_label.config(
                text="Sauvegarde et arrêt..."
            )

        except Exception as exc:
            messagebox.showerror(
                "Amber 0.1",
                str(exc)
            )

    def force_stop_v01_training(self):

        process = self.process

        if (
            process is None
            or process.poll() is not None
        ):
            self.v01_status_label.config(
                text="Aucun processus à forcer."
            )
            return

        if not messagebox.askyesno(
            "Amber 0.1",
            (
                "Forcer l'arrêt du processus ?\n\n"
                "À utiliser uniquement si Amber est bloqué avant "
                "un step ou pendant le chargement d'un checkpoint."
            )
        ):
            return

        try:
            process.kill()

            self.v01_status_label.config(
                text="Processus forcé à s'arrêter."
            )

            self._v01_log(
                "[V01] Processus arrêté de force."
            )

        except Exception as exc:
            messagebox.showerror(
                "Amber 0.1",
                str(exc)
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

        self.train_step_value.config(
            text=str(self.current_step)
        )

        if self.current_loss is None:
            self.train_loss_value.config(
                text="-"
            )
        else:
            self.train_loss_value.config(
                text=f"{float(self.current_loss):.4f}"
            )

        self.train_speed_value.config(
            text=f"{self.current_speed:.2f} step/s"
        )

        self.train_eta_value.config(
            text=self.current_eta
        )

        self.train_vram_value.config(
            text=f"{self.current_train_vram:.2f} GB"
        )

        self.train_governor_value.config(
            text=self.current_governor
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

            percent = (
                (self.current_step / target) * 100
                if target > 0
                else 0.0
            )

            self.progress_percent.config(
                text=f"{percent:.1f} %"
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
            self._timestamped_line(
                text
            )
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

            if (
                hasattr(
                    self,
                    "v01_console"
                )
                and (
                    "V01" in line
                    or "AUTOTUNE" in line
                    or "CACHE" in line
                    or "AMBER 0.1" in line
                    or "Parameters" in line
                    or "Tokenizer" in line
                )
            ):
                self._v01_log(
                    line
                )

            if line.startswith(
                "[V01 GEN JSON] "
            ):
                try:
                    payload = json.loads(
                        line.split(
                            " ",
                            3
                        )[3]
                    )

                    self.v01_generation_output.delete(
                        "1.0",
                        "end"
                    )

                    answer = payload.get(
                        "text",
                        ""
                    )

                    if not answer:
                        answer = (
                            "[Amber n'a produit aucun texte avant EOS]"
                        )

                    self.v01_generation_output.insert(
                        "1.0",
                        answer
                    )

                    self.v01_eval_status.config(
                        text=(
                            f"Génération : "
                            f"{int(payload.get('generated_tokens', 0))} tokens | "
                            f"{float(payload.get('tokens_per_second', 0.0)):.1f} tok/s"
                        )
                    )

                    self.refresh_v01_eval_status()

                except Exception as exc:
                    self.v01_eval_status.config(
                        text=f"Erreur lecture génération : {exc}"
                    )

            if line.startswith(
                "[V01 EVAL JSON] "
            ):
                try:
                    payload = json.loads(
                        line.split(
                            " ",
                            3
                        )[3]
                    )

                    loss = float(
                        payload.get(
                            "validation_loss"
                        )
                    )

                    perplexity = float(
                        payload.get(
                            "perplexity"
                        )
                    )

                    evaluated_tokens = int(
                        payload.get(
                            "evaluated_tokens",
                            0
                        )
                    )

                    self.eval_loss_value.config(
                        text=f"{loss:.4f}"
                    )

                    self.eval_ppl_value.config(
                        text=f"{perplexity:.2f}"
                    )

                    self.v01_eval_status.config(
                        text=(
                            f"{evaluated_tokens:,} tokens évalués | "
                            f"{float(payload.get('tokens_per_second', 0.0)):,.0f} tok/s"
                        )
                    )

                    self.refresh_v01_eval_status()

                except Exception as exc:
                    self.v01_eval_status.config(
                        text=f"Erreur lecture évaluation : {exc}"
                    )

            v01_match = re.search(
                (
                    r"\[V01 step=(\d+)\]\s+"
                    r"loss=([0-9.]+)\s+\|\s+"
                    r"val=([^\s]+)\s+\|\s+"
                    r"tokens=([0-9,]+)/([0-9,]+)\s+\|\s+"
                    r"speed=([0-9,]+)\s+tok/s\s+\|\s+"
                    r"lr=([^\s]+)\s+\|\s+"
                    r"vram=([0-9.]+)\s+GB\s+\|\s+"
                    r"(?:peak=([0-9.]+)\s+GB\s+\|\s+)?"
                    r"(?:reserved=([0-9.]+)\s+GB\s+\|\s+)?"
                    r"eta=([^\s]+)"
                ),
                line
            )

            if v01_match:
                step = int(
                    v01_match.group(1)
                )

                loss = float(
                    v01_match.group(2)
                )

                val_text = v01_match.group(3)

                tokens_seen = int(
                    v01_match.group(4).replace(
                        ",",
                        ""
                    )
                )

                target_tokens = int(
                    v01_match.group(5).replace(
                        ",",
                        ""
                    )
                )

                speed = int(
                    v01_match.group(6).replace(
                        ",",
                        ""
                    )
                )

                vram = float(
                    v01_match.group(8)
                )

                peak_vram = (
                    float(
                        v01_match.group(9)
                    )
                    if v01_match.group(9)
                    else vram
                )

                reserved_vram = (
                    float(
                        v01_match.group(10)
                    )
                    if v01_match.group(10)
                    else peak_vram
                )

                eta = v01_match.group(11)

                self.v01_seen_value.config(
                    text=format_tokens(
                        tokens_seen
                    )
                )

                self.v01_loss_value.config(
                    text=(
                        f"{loss:.3f} / "
                        f"{val_text}"
                    )
                )

                self.v01_progress.configure(
                    maximum=max(
                        target_tokens,
                        1
                    ),
                    value=min(
                        tokens_seen,
                        target_tokens
                    )
                )

                self.v01_progress_text.config(
                    text=(
                        f"{format_tokens(tokens_seen)} / "
                        f"{format_tokens(target_tokens)} tokens"
                    )
                )

                self.v01_metrics.config(
                    text=(
                        f"Step : {step:,} | "
                        f"Vitesse : {speed:,} tok/s | "
                        f"VRAM : {vram:.2f} GB instant. / "
                        f"{peak_vram:.2f} GB pic / "
                        f"{reserved_vram:.2f} GB réservée | "
                        f"ETA : {eta}"
                    )
                )

                self.v01_status_label.config(
                    text=(
                        f"Pré-entraînement — "
                        f"{tokens_seen / target_tokens * 100:.2f} %"
                    )
                )

            match = re.search(
                (
                    r"\[(\d{6})\]\s+"
                    r"loss=([0-9.]+)\s+\|\s+"
                    r"speed=([0-9.]+)\s+step/s"
                    r"(?:\s+\|\s+vram=([0-9.]+)\s+GB)?"
                    r"(?:\s+\|\s+eta=([^\s]+))?"
                ),
                line
            )

            if match:

                step = int(
                    match.group(1)
                )

                loss = float(
                    match.group(2)
                )

                speed = float(
                    match.group(3)
                )

                vram = (
                    float(match.group(4))
                    if match.group(4)
                    else self.current_train_vram
                )

                eta = (
                    match.group(5)
                    if match.group(5)
                    else self.current_eta
                )

                self.current_step = step
                self.current_loss = loss
                self.current_speed = speed
                self.current_train_vram = vram
                self.current_eta = eta
                self.current_governor = self.mode.get()

                self.step_value.config(
                    text=str(
                        step
                    )
                )

                self.loss_value.config(
                    text=f"{loss:.4f}"
                )

                self.train_step_value.config(
                    text=str(step)
                )

                self.train_loss_value.config(
                    text=f"{loss:.4f}"
                )

                self.train_speed_value.config(
                    text=f"{speed:.2f} step/s"
                )

                self.train_vram_value.config(
                    text=f"{vram:.2f} GB"
                )

                self.train_eta_value.config(
                    text=eta
                )

                self.train_governor_value.config(
                    text=self.current_governor
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

                    percent = (
                        (step / target) * 100
                        if target > 0
                        else 0.0
                    )

                    self.progress_percent.config(
                        text=f"{percent:.1f} %"
                    )

                except Exception:
                    pass

            if (
                "[Checkpoint]"
                in line
            ):
                self.refresh_checkpoints()

            if "Game detected:" in line:
                self.current_governor = "GAME PAUSE"
                self.train_governor_value.config(
                    text="GAME PAUSE"
                )

            if (
                "Training paused"
                in line
            ):
                self.training_status.config(
                    text="En pause"
                )

                self.current_governor = (
                    "GAME PAUSE"
                    if "[Governor]" in line
                    else "PAUSE"
                )

                self.train_governor_value.config(
                    text=self.current_governor
                )

            if (
                "Training resumed"
                in line
            ):
                self.training_status.config(
                    text="Entraînement en cours..."
                )

                self.current_governor = self.mode.get()

                self.train_governor_value.config(
                    text=self.current_governor
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

            if "[AUTOTUNE BEST]" in line:
                self.v01_status_label.config(
                    text="Auto-Tuner terminé — profil FULL prêt."
                )
                self.refresh_v01_status()

            if "[CACHE COMPLETE]" in line:
                self.v01_status_label.config(
                    text="Cache tokens construit."
                )
                self.refresh_v01_status()

            if "[V01] En pause." in line:
                self.v01_status_label.config(
                    text="En pause"
                )

            if "[V01] Reprise." in line:
                self.v01_status_label.config(
                    text="Pré-entraînement en cours..."
                )

            if (
                "[V01] Arrêt propre terminé."
                in line
            ):
                self.v01_status_label.config(
                    text="Arrêté proprement"
                )
                self.refresh_v01_status()

            if (
                "AMBER 0.1 PRETRAINING STOP TARGET REACHED"
                in line
            ):
                self.v01_status_label.config(
                    text="Objectif atteint"
                )
                self.refresh_v01_status()

            if (
                "[Process finished:"
                in line
            ):
                self.after(
                    300,
                    self.refresh_status
                )

                if hasattr(
                    self,
                    "v01_status_label"
                ):
                    self.after(
                        500,
                        self.refresh_v01_status
                    )

                if hasattr(
                    self,
                    "eval_checkpoint_value"
                ):
                    self.after(
                        600,
                        self.refresh_v01_eval_status
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

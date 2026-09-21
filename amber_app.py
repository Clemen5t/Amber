import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk

from pathlib import Path
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText

import torch

from amber.model import AmberConfig, AmberModel
from amber.dataset_manager import (
    SOURCE_CATALOG,
    PREPARED_TRAIN_FILE,
    PREPARED_VALIDATION_FILE,
    dataset_stats,
    file_stats,
    format_bytes,
    import_local_file,
    download_source,
    download_catalog_source,
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


class AmberApp(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title(
            "Amber 0.0.6"
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
        self.current_speed = 0.0
        self.current_eta = "--:--"
        self.current_train_vram = 0.0
        self.current_governor = "BALANCED"

        self.paused = False
        self.dataset_busy = False

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
            text="AI Control Center · Amber Model 0.0.6",
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

        self._build_dashboard()
        self._build_chat()
        self._build_training()
        self._build_checkpoints()
        self._build_dataset()

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
            "Amber Control Center 0.0.6 ready."
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

        self.dataset_catalog = tk.StringVar(
            value=next(iter(SOURCE_CATALOG.keys()))
        )

        self.tokenizer_vocab_size = tk.StringVar(
            value="32000"
        )

        ttk.Label(
            self.dataset_tab,
            text="Dataset Manager 0.0.6",
            style="Title.TLabel"
        ).pack(
            anchor="w",
            pady=(20, 5)
        )

        ttk.Label(
            self.dataset_tab,
            text=(
                "Collecte, nettoyage, déduplication, split train/validation "
                "et entraînement du tokenizer Amber."
            ),
            style="Subtitle.TLabel"
        ).pack(
            anchor="w",
            pady=(0, 15)
        )

        cards = ttk.Frame(
            self.dataset_tab
        )

        cards.pack(
            fill="x",
            pady=(0, 14)
        )

        self.dataset_seed_value = self._card(
            cards,
            "SEED"
        )

        self.dataset_sources_value = self._card(
            cards,
            "SOURCES"
        )

        self.dataset_train_value = self._card(
            cards,
            "TRAIN PRÉPARÉ"
        )

        self.dataset_validation_value = self._card(
            cards,
            "VALIDATION"
        )

        self.dataset_tokenizer_value = self._card(
            cards,
            "TOKENIZER"
        )

        source_box = ttk.Frame(
            self.dataset_tab
        )

        source_box.pack(
            fill="x",
            pady=6
        )

        ttk.Button(
            source_box,
            text="Importer fichiers",
            command=self.import_dataset_files
        ).pack(
            side="left",
            padx=(0, 8)
        )

        catalog_values = [
            key
            for key in SOURCE_CATALOG.keys()
        ]

        self.catalog_combo = ttk.Combobox(
            source_box,
            textvariable=self.dataset_catalog,
            values=catalog_values,
            state="readonly",
            width=24
        )

        self.catalog_combo.pack(
            side="left",
            padx=8
        )

        ttk.Button(
            source_box,
            text="Télécharger preset",
            command=self.download_catalog_dataset
        ).pack(
            side="left",
            padx=8
        )

        custom = ttk.Frame(
            self.dataset_tab
        )

        custom.pack(
            fill="x",
            pady=6
        )

        ttk.Entry(
            custom,
            textvariable=self.dataset_url
        ).pack(
            side="left",
            fill="x",
            expand=True,
            padx=(0, 8)
        )

        ttk.Button(
            custom,
            text="Télécharger URL",
            command=self.download_custom_dataset
        ).pack(
            side="left"
        )

        pipeline = ttk.Frame(
            self.dataset_tab
        )

        pipeline.pack(
            fill="x",
            pady=(12, 6)
        )

        ttk.Button(
            pipeline,
            text="Préparer dataset",
            command=self.prepare_dataset_ui
        ).pack(
            side="left",
            padx=(0, 8)
        )

        ttk.Label(
            pipeline,
            text="Vocab :"
        ).pack(
            side="left",
            padx=(12, 4)
        )

        ttk.Entry(
            pipeline,
            textvariable=self.tokenizer_vocab_size,
            width=10
        ).pack(
            side="left",
            padx=(0, 8)
        )

        ttk.Button(
            pipeline,
            text="Entraîner tokenizer",
            command=self.train_tokenizer_ui
        ).pack(
            side="left",
            padx=8
        )

        ttk.Button(
            pipeline,
            text="Créer manifest",
            command=self.create_dataset_manifest
        ).pack(
            side="left",
            padx=8
        )

        ttk.Button(
            pipeline,
            text="Ouvrir data",
            command=lambda: os.startfile(
                DATA_DIR
            )
        ).pack(
            side="left",
            padx=8
        )

        self.dataset_progress_label = ttk.Label(
            self.dataset_tab,
            text="Prêt"
        )

        self.dataset_progress_label.pack(
            anchor="w",
            pady=(8, 4)
        )

        lower = ttk.Frame(
            self.dataset_tab
        )

        lower.pack(
            fill="both",
            expand=True,
            pady=(4, 0)
        )

        left = ttk.Frame(
            lower
        )

        left.pack(
            side="left",
            fill="both",
            expand=False,
            padx=(0, 10)
        )

        ttk.Label(
            left,
            text="Sources locales"
        ).pack(
            anchor="w",
            pady=(0, 4)
        )

        self.dataset_source_list = tk.Listbox(
            left,
            bg="#0b0b0f",
            fg="#ffffff",
            relief="flat",
            font=("Consolas", 9),
            width=43
        )

        self.dataset_source_list.pack(
            fill="both",
            expand=True
        )

        right = ttk.Frame(
            lower
        )

        right.pack(
            side="left",
            fill="both",
            expand=True
        )

        ttk.Label(
            right,
            text="Journal Dataset"
        ).pack(
            anchor="w",
            pady=(0, 4)
        )

        self.dataset_status = ScrolledText(
            right,
            bg="#0b0b0f",
            fg="#e5e5ea",
            insertbackground="white",
            relief="flat",
            font=("Consolas", 9),
            height=15
        )

        self.dataset_status.pack(
            fill="both",
            expand=True
        )

        self.refresh_dataset()

    def _dataset_log(
        self,
        text
    ):
        self.dataset_status.insert(
            "end",
            str(text) + "\n"
        )

        self.dataset_status.see(
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
                "Vérifie que tu as le droit de l'utiliser "
                "pour l'entraînement."
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
            "Téléchargement...",
            lambda: download_source(
                url,
                progress_callback=progress
            ),
            lambda result: (
                f"Téléchargé : {result['name']} "
                f"({result['size_human']})"
            )
        )

    def download_catalog_dataset(self):

        key = self.dataset_catalog.get()

        source = SOURCE_CATALOG.get(
            key
        )

        if not source:
            return

        details = (
            f"{source['name']}\n\n"
            f"{source.get('note', '')}\n\n"
            f"Licence indiquée : "
            f"{source.get('license', 'à vérifier')}\n\n"
            "Ce téléchargement peut être très volumineux. Continuer ?"
        )

        if not messagebox.askyesno(
            "Amber Dataset Manager",
            details
        ):
            return

        def progress(
            downloaded,
            total
        ):
            if total:
                text = (
                    f"Téléchargement preset : "
                    f"{format_bytes(downloaded)} / "
                    f"{format_bytes(total)} "
                    f"({downloaded / total * 100:.1f} %)"
                )
            else:
                text = (
                    "Téléchargement preset : "
                    f"{format_bytes(downloaded)}"
                )

            self.after(
                0,
                lambda value=text: self._dataset_progress(
                    value
                )
            )

        self._run_dataset_task(
            "Téléchargement preset...",
            lambda: download_catalog_source(
                key,
                progress_callback=progress
            ),
            lambda result: (
                f"Preset téléchargé : {result['name']} "
                f"({result['size_human']})"
            )
        )

    def prepare_dataset_ui(self):

        if not messagebox.askyesno(
            "Amber Dataset Manager",
            (
                "Préparer le dataset ?\n\n"
                "Amber va nettoyer les textes, supprimer les doublons "
                "et créer un split train/validation 98/2."
            )
        ):
            return

        def progress(
            info
        ):
            stage = info.get(
                "stage",
                ""
            )

            if stage == "source":
                text = (
                    f"Préparation : {info.get('source')} "
                    f"({info.get('source_index')}/"
                    f"{info.get('source_count')})"
                )
            else:
                text = (
                    "Préparation : "
                    f"{info.get('train_docs', 0):,} train / "
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
                progress_callback=progress
            ),
            lambda result: (
                "Dataset prêt : "
                f"{result['train_docs']:,} docs train, "
                f"{result['validation_docs']:,} docs validation, "
                f"{result['skipped']:,} ignorés/doublons."
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

    def create_dataset_manifest(self):

        try:
            manifest = write_manifest()

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

        seed = dataset_stats()
        sources = list_raw_sources()
        prepared_train = file_stats(
            PREPARED_TRAIN_FILE
        )
        prepared_validation = file_stats(
            PREPARED_VALIDATION_FILE
        )
        tok = tokenizer_status()

        self.dataset_seed_value.config(
            text=seed["size_human"]
            if seed["exists"]
            else "-"
        )

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

        self.dataset_source_list.delete(
            0,
            "end"
        )

        if not sources:
            self.dataset_source_list.insert(
                "end",
                "Aucune source dans data/raw"
            )
        else:
            for source in sources:
                self.dataset_source_list.insert(
                    "end",
                    (
                        f"{source['name']}  "
                        f"[{source['size_human']}]"
                    )
                )

        self.dataset_status.delete(
            "1.0",
            "end"
        )

        self._dataset_log(
            "AMBER DATASET MANAGER 0.0.6"
        )

        self._dataset_log(
            "=" * 52
        )

        if seed["exists"]:
            self._dataset_log(
                (
                    f"Seed historique : "
                    f"{seed['characters']:,} caractères "
                    f"({seed['size_human']})"
                )
            )
        else:
            self._dataset_log(
                "Seed historique : absent"
            )

        self._dataset_log(
            (
                f"Sources brutes : {len(sources)} "
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
            "Ordre recommandé :"
        )

        self._dataset_log(
            "1. Importer/télécharger des sources autorisées"
        )

        self._dataset_log(
            "2. Préparer dataset"
        )

        self._dataset_log(
            "3. Entraîner tokenizer"
        )

        self._dataset_log(
            "4. Amber 0.1 utilisera ces artefacts"
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

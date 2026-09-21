import json
import os
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox


ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "version.json"
LOCAL_CONFIG = ROOT / "updater_local.json"
BACKUP_ROOT = ROOT / "backups"

SOURCE_TARGETS = [
    "amber",
    "training",
    "tokenizer",
    "inference",
    "tests",
    "amber_app.py",
    "version.json",
    ".gitignore",
    "requirements.txt",
    "data/train.txt",
]


def read_version():
    try:
        data = json.loads(VERSION_FILE.read_text(encoding="utf-8"))
        return str(data.get("version", "0.0.0"))
    except Exception:
        return "0.0.0"


CURRENT_VERSION = read_version()


def _creation_flags():
    if os.name != "nt":
        return 0
    return subprocess.CREATE_NO_WINDOW


def _run(args, cwd=ROOT, timeout=120):
    return subprocess.run(
        args,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=_creation_flags(),
        timeout=timeout,
        check=False,
    )


def _git(*args, timeout=120):
    return _run(["git", *args], timeout=timeout)


def git_available():
    try:
        result = _git("--version", timeout=10)
        return result.returncode == 0
    except Exception:
        return False


def is_git_repo():
    return (ROOT / ".git").exists()


def current_branch():
    if not is_git_repo():
        return None
    result = _git("branch", "--show-current", timeout=15)
    if result.returncode != 0:
        return None
    return result.stdout.strip() or "main"


def remote_url():
    if not is_git_repo():
        return None
    result = _git("remote", "get-url", "origin", timeout=15)
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def load_local_config():
    try:
        return json.loads(LOCAL_CONFIG.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_local_config(data):
    LOCAL_CONFIG.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8"
    )


def git_status():
    status = {
        "git": git_available(),
        "repo": False,
        "remote": None,
        "branch": None,
        "update_available": False,
        "behind": 0,
        "message": "",
    }

    if not status["git"]:
        status["message"] = "Git n'est pas installé."
        return status

    status["repo"] = is_git_repo()

    if not status["repo"]:
        status["message"] = "Amber n'est pas encore relié à GitHub."
        return status

    status["branch"] = current_branch() or "main"
    status["remote"] = remote_url()

    if not status["remote"]:
        status["message"] = "Dépôt local présent, mais aucun remote origin."
        return status

    fetch = _git("fetch", "origin", "--quiet", timeout=90)
    if fetch.returncode != 0:
        status["message"] = (
            "Impossible de contacter GitHub.\n" + fetch.stdout.strip()
        )
        return status

    branch = status["branch"]
    remote_ref = f"origin/{branch}"

    exists = _git("rev-parse", "--verify", remote_ref, timeout=15)
    if exists.returncode != 0 and branch != "main":
        branch = "main"
        remote_ref = "origin/main"
        exists = _git("rev-parse", "--verify", remote_ref, timeout=15)

    if exists.returncode != 0:
        status["message"] = (
            "Le dépôt distant ne contient pas encore de branche compatible."
        )
        return status

    behind = _git("rev-list", "--count", f"HEAD..{remote_ref}", timeout=15)

    if behind.returncode == 0:
        try:
            status["behind"] = int(behind.stdout.strip() or "0")
        except ValueError:
            status["behind"] = 0

    status["update_available"] = status["behind"] > 0

    if status["update_available"]:
        status["message"] = f"{status['behind']} mise(s) à jour disponible(s)."
    else:
        status["message"] = f"Amber {CURRENT_VERSION} est à jour."

    return status


def initialize_repo(remote):
    if not git_available():
        return False, "Git n'est pas installé."

    remote = remote.strip()
    if not remote:
        return False, "URL GitHub manquante."

    if not (
        remote.startswith("https://github.com/")
        or remote.startswith("git@github.com:")
    ):
        return False, "Utilise une URL de dépôt GitHub."

    if not is_git_repo():
        init = _git("init", "-b", "main", timeout=30)
        if init.returncode != 0:
            init = _git("init", timeout=30)
            if init.returncode != 0:
                return False, init.stdout.strip()
            _git("branch", "-M", "main", timeout=30)

    existing = remote_url()

    if existing:
        set_url = _git("remote", "set-url", "origin", remote, timeout=30)
        if set_url.returncode != 0:
            return False, set_url.stdout.strip()
    else:
        add = _git("remote", "add", "origin", remote, timeout=30)
        if add.returncode != 0:
            return False, add.stdout.strip()

    cfg = load_local_config()
    cfg["remote"] = remote
    save_local_config(cfg)

    return True, "Dépôt GitHub configuré."


def _ensure_git_identity():
    name = _git("config", "user.name", timeout=10)
    email = _git("config", "user.email", timeout=10)

    if (
        name.returncode == 0
        and name.stdout.strip()
        and email.returncode == 0
        and email.stdout.strip()
    ):
        return True, ""

    return False, (
        "Git n'a pas encore de nom/email configuré.\n\n"
        "Configure une fois ton identité Git avant de publier la base."
    )


def publish_baseline():
    if not is_git_repo():
        return False, "Configure d'abord le dépôt GitHub."

    if not remote_url():
        return False, "Aucun remote origin configuré."

    ok, msg = _ensure_git_identity()
    if not ok:
        return False, msg

    paths = []
    for rel in SOURCE_TARGETS:
        if (ROOT / rel).exists():
            paths.append(rel)

    add = _git("add", "--", *paths, timeout=90)
    if add.returncode != 0:
        return False, add.stdout.strip()

    _git("add", ".gitignore", timeout=30)

    status = _git("status", "--porcelain", timeout=30)
    if status.returncode != 0:
        return False, status.stdout.strip()

    if status.stdout.strip():
        commit = _git(
            "commit",
            "-m",
            f"Amber {CURRENT_VERSION} baseline",
            timeout=90
        )
        if commit.returncode != 0:
            return False, commit.stdout.strip()

    branch = current_branch() or "main"

    push = _git(
        "push",
        "-u",
        "origin",
        branch,
        timeout=180
    )

    if push.returncode != 0:
        return False, (
            "Le push GitHub a échoué.\n\n"
            + push.stdout.strip()
            + "\n\nUne authentification GitHub peut être nécessaire une seule fois."
        )

    return True, f"Amber {CURRENT_VERSION} a été publié sur GitHub."


def tracked_worktree_dirty():
    if not is_git_repo():
        return False, ""

    result = _git("status", "--porcelain", timeout=30)
    if result.returncode != 0:
        return True, result.stdout.strip()

    dirty_lines = []

    for line in result.stdout.splitlines():
        if not line.strip():
            continue

        # Les fichiers non suivis (ex: Amber.bat ou anciens setup)
        # ne doivent pas bloquer une mise à jour du code suivi.
        if line.startswith("??"):
            continue

        dirty_lines.append(line)

    if dirty_lines:
        return True, "\n".join(dirty_lines[:20])

    return False, ""


def create_backup():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = BACKUP_ROOT / f"before_update_{stamp}"
    destination.mkdir(parents=True, exist_ok=True)

    for rel in SOURCE_TARGETS:
        source = ROOT / rel
        if not source.exists():
            continue

        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)

        if source.is_dir():
            shutil.copytree(
                source,
                target,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
            )
        else:
            shutil.copy2(source, target)

    return destination


def latest_backup():
    if not BACKUP_ROOT.exists():
        return None

    candidates = sorted(
        [
            p for p in BACKUP_ROOT.iterdir()
            if p.is_dir() and p.name.startswith("before_update_")
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    return candidates[0] if candidates else None


def restore_backup(path):
    for rel in SOURCE_TARGETS:
        source = path / rel
        if not source.exists():
            continue

        target = ROOT / rel

        if source.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def install_update():
    if not is_git_repo() or not remote_url():
        return False, "GitHub n'est pas encore configuré.", None

    dirty, details = tracked_worktree_dirty()

    if dirty:
        return (
            False,
            "Des fichiers source locaux ont été modifiés.\n"
            "Amber refuse de les écraser automatiquement.\n\n"
            + details,
            None,
        )

    backup = create_backup()

    fetch = _git("fetch", "origin", "--quiet", timeout=90)
    if fetch.returncode != 0:
        return False, fetch.stdout.strip(), backup

    branch = current_branch() or "main"

    pull = _git(
        "pull",
        "--ff-only",
        "origin",
        branch,
        timeout=180
    )

    if pull.returncode != 0:
        return False, pull.stdout.strip(), backup

    requirements = ROOT / "requirements.txt"

    if requirements.exists():
        pip = _run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                str(requirements),
            ],
            timeout=600
        )

        if pip.returncode != 0:
            return (
                False,
                "Code mis à jour, mais dépendances en erreur:\n"
                + pip.stdout.strip(),
                backup,
            )

    return True, "Mise à jour installée.", backup


def restart_amber(parent):
    args = [sys.executable, str(ROOT / "amber_app.py")]

    kwargs = {
        "cwd": ROOT,
        "close_fds": True,
    }

    if os.name == "nt":
        kwargs["creationflags"] = (
            subprocess.CREATE_NO_WINDOW
            | subprocess.DETACHED_PROCESS
        )

    subprocess.Popen(args, **kwargs)
    parent.after(150, parent.destroy)


class UpdaterWindow(tk.Toplevel):
    def __init__(self, parent, process_getter=None):
        super().__init__(parent)

        self.parent = parent
        self.process_getter = process_getter or (lambda: None)
        self.title("Amber Updater")
        self.geometry("700x520")
        self.minsize(650, 480)
        self.configure(bg="#101014")
        self.transient(parent)

        cfg = load_local_config()
        existing_remote = remote_url() or cfg.get("remote", "")

        self.remote_var = tk.StringVar(value=existing_remote)
        self.status_var = tk.StringVar(value="Prêt.")
        self.version_var = tk.StringVar(value=f"Version {read_version()}")

        self._build()
        self.after(250, self.check_updates)

    def _build(self):
        body = tk.Frame(self, bg="#101014")
        body.pack(fill="both", expand=True, padx=22, pady=20)

        tk.Label(
            body,
            text="AMBER UPDATER",
            bg="#101014",
            fg="white",
            font=("Segoe UI Semibold", 22)
        ).pack(anchor="w")

        tk.Label(
            body,
            textvariable=self.version_var,
            bg="#101014",
            fg="#ffb347",
            font=("Segoe UI Semibold", 11)
        ).pack(anchor="w", pady=(2, 16))

        status_box = tk.Frame(body, bg="#191920")
        status_box.pack(fill="x", pady=(0, 16))

        tk.Label(
            status_box,
            textvariable=self.status_var,
            bg="#191920",
            fg="white",
            justify="left",
            anchor="w",
            wraplength=620,
            font=("Segoe UI", 10)
        ).pack(fill="x", padx=14, pady=14)

        tk.Label(
            body,
            text="Dépôt GitHub privé",
            bg="#101014",
            fg="white",
            font=("Segoe UI Semibold", 10)
        ).pack(anchor="w")

        self.remote_entry = tk.Entry(
            body,
            textvariable=self.remote_var,
            bg="#191920",
            fg="white",
            insertbackground="white",
            relief="flat",
            font=("Segoe UI", 10)
        )
        self.remote_entry.pack(fill="x", ipady=8, pady=(6, 10))

        row1 = tk.Frame(body, bg="#101014")
        row1.pack(fill="x", pady=4)

        self._button(
            row1,
            "Configurer GitHub",
            self.configure_repo
        ).pack(side="left", padx=(0, 8))

        self._button(
            row1,
            f"Publier Amber {CURRENT_VERSION}",
            self.publish
        ).pack(side="left", padx=8)

        row2 = tk.Frame(body, bg="#101014")
        row2.pack(fill="x", pady=(16, 4))

        self._button(
            row2,
            "Vérifier les mises à jour",
            self.check_updates
        ).pack(side="left", padx=(0, 8))

        self._button(
            row2,
            "Installer la mise à jour",
            self.install
        ).pack(side="left", padx=8)

        self._button(
            row2,
            "Restaurer dernière sauvegarde",
            self.rollback
        ).pack(side="left", padx=8)

        tk.Label(
            body,
            text=(
                "Checkpoints, .venv et futurs gros datasets restent locaux. "
                "L'Updater travaille silencieusement et ne lance pas de fenêtres CMD."
            ),
            bg="#101014",
            fg="#9c9ca8",
            wraplength=630,
            justify="left",
            font=("Segoe UI", 9)
        ).pack(anchor="w", pady=(22, 0))

    def _button(self, parent, text, command):
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg="#2b2b34",
            fg="white",
            activebackground="#3a3a46",
            activeforeground="white",
            relief="flat",
            padx=12,
            pady=8,
            font=("Segoe UI Semibold", 9)
        )

    def busy_training(self):
        process = self.process_getter()
        return process is not None and process.poll() is None

    def run_async(self, fn, done):
        self.configure(cursor="watch")

        def worker():
            try:
                result = fn()
                self.after(0, lambda: done(result))
            except Exception as exc:
                self.after(0, lambda: self._error(str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _error(self, message):
        self.configure(cursor="")
        self.status_var.set(message)
        messagebox.showerror("Amber Updater", message, parent=self)

    def configure_repo(self):
        remote = self.remote_var.get().strip()
        self.status_var.set("Configuration de GitHub...")
        self.run_async(
            lambda: initialize_repo(remote),
            self._configured
        )

    def _configured(self, result):
        self.configure(cursor="")
        ok, message = result
        self.status_var.set(message)
        if ok:
            messagebox.showinfo("Amber Updater", message, parent=self)

    def publish(self):
        if self.busy_training():
            messagebox.showwarning(
                "Amber Updater",
                "Arrête l'entraînement avant de publier.",
                parent=self
            )
            return

        self.status_var.set(
            f"Publication de la base Amber {CURRENT_VERSION}..."
        )
        self.run_async(publish_baseline, self._published)

    def _published(self, result):
        self.configure(cursor="")
        ok, message = result
        self.status_var.set(message)
        if ok:
            messagebox.showinfo("Amber Updater", message, parent=self)

    def check_updates(self):
        self.status_var.set("Vérification silencieuse...")
        self.run_async(git_status, self._checked)

    def _checked(self, status):
        self.configure(cursor="")
        self.status_var.set(status["message"])
        if status.get("remote"):
            self.remote_var.set(status["remote"])

    def install(self):
        if self.busy_training():
            messagebox.showwarning(
                "Amber Updater",
                "Sauvegarde et arrête l'entraînement avant la mise à jour.",
                parent=self
            )
            return

        answer = messagebox.askyesno(
            "Amber Updater",
            (
                "Installer la mise à jour disponible ?\n\n"
                "Une sauvegarde du code actuel sera créée "
                "et Amber redémarrera automatiquement."
            ),
            parent=self
        )

        if not answer:
            return

        self.status_var.set("Installation en cours...")
        self.run_async(install_update, self._installed)

    def _installed(self, result):
        self.configure(cursor="")
        ok, message, backup = result
        self.status_var.set(message)

        if not ok:
            messagebox.showerror("Amber Updater", message, parent=self)
            return

        messagebox.showinfo(
            "Amber Updater",
            "Mise à jour terminée. Amber va redémarrer.",
            parent=self
        )
        restart_amber(self.parent)

    def rollback(self):
        if self.busy_training():
            messagebox.showwarning(
                "Amber Updater",
                "Arrête l'entraînement avant une restauration.",
                parent=self
            )
            return

        backup = latest_backup()

        if backup is None:
            messagebox.showinfo(
                "Amber Updater",
                "Aucune sauvegarde de mise à jour disponible.",
                parent=self
            )
            return

        if not messagebox.askyesno(
            "Amber Updater",
            f"Restaurer {backup.name} ?",
            parent=self
        ):
            return

        try:
            restore_backup(backup)
        except Exception as exc:
            self._error(str(exc))
            return

        messagebox.showinfo(
            "Amber Updater",
            "Sauvegarde restaurée. Amber va redémarrer.",
            parent=self
        )
        restart_amber(self.parent)


def open_updater_window(parent, process_getter=None):
    UpdaterWindow(parent, process_getter=process_getter)

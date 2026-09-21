import ctypes
import os
import subprocess
import time


class LASTINPUTINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_uint),
        ("dwTime", ctypes.c_uint),
    ]


class TrainingGovernor:
    GAMES = [
        "fortniteclient-win64-shipping.exe",
        "rocketleague.exe",
        "valorant-win64-shipping.exe",
        "valorant.exe",
        "r5apex.exe",
        "overwatch.exe",
        "cs2.exe",
        "cod.exe",
    ]

    def __init__(self, mode="balanced"):
        mode = mode.lower()

        if mode not in {
            "eco",
            "balanced",
            "full"
        }:
            mode = "balanced"

        self.mode = mode

    def idle_seconds(self):
        try:
            info = LASTINPUTINFO()

            info.cbSize = ctypes.sizeof(
                info
            )

            ctypes.windll.user32.GetLastInputInfo(
                ctypes.byref(info)
            )

            elapsed = (
                ctypes.windll.kernel32.GetTickCount()
                - info.dwTime
            )

            return elapsed / 1000.0

        except Exception:
            return 999.0

    def active_game(self):
        try:
            creation_flags = 0

            if os.name == "nt":
                creation_flags = subprocess.CREATE_NO_WINDOW

            output = subprocess.check_output(
                [
                    "tasklist",
                    "/FO",
                    "CSV",
                    "/NH"
                ],
                text=True,
                errors="ignore",
                creationflags=creation_flags
            ).lower()

            for game in self.GAMES:
                if game in output:
                    return game

        except Exception:
            pass

        return None

    def throttle(self):
        if self.mode == "full":
            return

        idle = self.idle_seconds()

        if self.mode == "eco":
            if idle < 60:
                time.sleep(0.20)
            else:
                time.sleep(0.08)

        elif self.mode == "balanced":
            if idle < 30:
                time.sleep(0.05)
            else:
                time.sleep(0.01)

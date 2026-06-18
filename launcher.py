"""
RL Launcher — GUI pentru CarRacing si Pong
"""
import tkinter as tk
from tkinter import ttk, scrolledtext
import subprocess
import threading
import sys
import os


# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARRACING_DIR = BASE_DIR
PONG_DIR = os.path.join(BASE_DIR, "pong")

SCRIPTS = {
    "carracing_train": os.path.join(CARRACING_DIR, "train_gpu.py"),
    "carracing_play":  os.path.join(CARRACING_DIR, "test_trained.py"),
    "pong_train":      os.path.join(PONG_DIR,      "train.py"),
    "pong_play":       os.path.join(PONG_DIR,      "play.py"),
    "pong_watch":      os.path.join(PONG_DIR,      "watch.py"),
}

# ── Palette ──────────────────────────────────────────────────────────────────
BG          = "#0f0f13"
PANEL       = "#16161d"
BORDER      = "#2a2a38"
ACCENT_CAR  = "#e8a020"   # amber  — CarRacing
ACCENT_PONG = "#3db8f5"   # cyan   — Pong
TEXT        = "#e0e0e8"
TEXT_DIM    = "#6b6b82"
GREEN       = "#4caf7d"
RED         = "#e05c5c"
FONT_MONO   = ("Consolas", 9)
FONT_UI     = ("Segoe UI", 9)
FONT_TITLE  = ("Segoe UI Semibold", 11)
FONT_HEAD   = ("Segoe UI Semibold", 9)


# ── Process manager ──────────────────────────────────────────────────────────
class ProcessManager:
    def __init__(self):
        self._procs: dict[str, subprocess.Popen] = {}

    def start(self, key: str, script: str, log_widget, status_cb):
        if key in self._procs and self._procs[key].poll() is None:
            return  # already running

        def run():
            try:
                proc = subprocess.Popen(
                    [sys.executable, "-u", script],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=os.path.dirname(script),
                )
                self._procs[key] = proc
                status_cb("running")
                for line in proc.stdout:
                    log_widget.after(0, _append, log_widget, line)
                proc.wait()
                status_cb("stopped" if proc.returncode == 0 else "error")
            except Exception as e:
                log_widget.after(0, _append, log_widget, f"[ERROR] {e}\n")
                status_cb("error")

        threading.Thread(target=run, daemon=True).start()

    def stop(self, key: str):
        proc = self._procs.get(key)
        if proc and proc.poll() is None:
            proc.terminate()

    def is_running(self, key: str) -> bool:
        proc = self._procs.get(key)
        return proc is not None and proc.poll() is None


def _append(widget: scrolledtext.ScrolledText, text: str):
    widget.configure(state="normal")
    widget.insert(tk.END, text)
    widget.see(tk.END)
    widget.configure(state="disabled")


# ── UI helpers ───────────────────────────────────────────────────────────────
def sep(parent, color=BORDER):
    tk.Frame(parent, height=1, bg=color).pack(fill="x", pady=(8, 8))


def label(parent, text, font=FONT_UI, color=TEXT, **kw):
    tk.Label(parent, text=text, font=font, fg=color, bg=PANEL, **kw).pack(**kw)


def status_dot(parent):
    c = tk.Canvas(parent, width=10, height=10, bg=PANEL, highlightthickness=0)
    dot = c.create_oval(1, 1, 9, 9, fill=TEXT_DIM, outline="")
    c.pack(side="left", padx=(0, 6))
    return c, dot


def make_btn(parent, text, accent, command):
    b = tk.Button(
        parent, text=text, font=FONT_HEAD,
        fg=BG, bg=accent, activeforeground=BG, activebackground=accent,
        relief="flat", bd=0, padx=12, pady=5,
        cursor="hand2", command=command,
    )
    b.pack(side="left", padx=(0, 6))
    return b


# ── Card ─────────────────────────────────────────────────────────────────────
class Card(tk.Frame):
    def __init__(self, parent, title: str, accent: str, actions: list[dict],
                 proc_mgr: ProcessManager, shared_log):
        super().__init__(parent, bg=PANEL, bd=0,
                         highlightthickness=1, highlightbackground=BORDER)
        self.accent = accent
        self.proc_mgr = proc_mgr
        self.shared_log = shared_log
        self.action_defs = actions
        self._build(title, actions)

    def _build(self, title: str, actions: list[dict]):
        # accent bar
        tk.Frame(self, height=3, bg=self.accent).pack(fill="x")

        inner = tk.Frame(self, bg=PANEL)
        inner.pack(fill="both", expand=True, padx=14, pady=10)

        # title row
        title_row = tk.Frame(inner, bg=PANEL)
        title_row.pack(fill="x")
        tk.Label(title_row, text=title, font=FONT_TITLE,
                 fg=TEXT, bg=PANEL).pack(side="left")

        # status indicator
        self._dot_canvas, self._dot = status_dot(title_row)
        self._dot_canvas.pack(side="right")
        self._status_lbl = tk.Label(title_row, text="idle",
                                    font=FONT_UI, fg=TEXT_DIM, bg=PANEL)
        self._status_lbl.pack(side="right", padx=(0, 4))

        sep(inner)

        # buttons
        btn_row = tk.Frame(inner, bg=PANEL)
        btn_row.pack(fill="x")
        for a in actions:
            make_btn(btn_row, a["label"], self.accent,
                     lambda key=a["key"], script=a["script"]: self._launch(key, script))

        stop_btn = tk.Button(
            btn_row, text="■ Stop", font=FONT_HEAD,
            fg=RED, bg=PANEL, activeforeground=RED, activebackground=PANEL,
            relief="flat", bd=0, padx=8, pady=5,
            cursor="hand2", command=self._stop_all,
        )
        stop_btn.pack(side="right")

    def _launch(self, key, script):
        if not os.path.exists(script):
            self._set_status("missing", RED)
            _append(self.shared_log,
                    f"[ERROR] Script not found: {script}\n")
            return
        self._set_status("starting…", self.accent)
        self.proc_mgr.start(key, script, self.shared_log,
                            lambda s, k=key: self._on_status(s, k))

    def _stop_all(self):
        for a in self.action_defs:
            self.proc_mgr.stop(a["key"])
        self._set_status("stopped", TEXT_DIM)

    def _on_status(self, status: str, key: str):
        colors = {"running": GREEN, "stopped": TEXT_DIM, "error": RED}
        self._set_status(status, colors.get(status, TEXT_DIM))

    def _set_status(self, text: str, color: str):
        dot_colors = {GREEN: GREEN, RED: RED, TEXT_DIM: TEXT_DIM,
                      ACCENT_CAR: ACCENT_CAR, ACCENT_PONG: ACCENT_PONG}
        self._status_lbl.after(0, lambda: (
            self._status_lbl.configure(text=text, fg=color),
            self._dot_canvas.itemconfig(self._dot, fill=color),
        ))


# ── Main window ──────────────────────────────────────────────────────────────
class Launcher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RL Launcher")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(720, 560)

        pm = ProcessManager()
        self._build(pm)

    def _build(self, pm: ProcessManager):
        # ── header
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=20, pady=(16, 0))
        tk.Label(hdr, text="RL Launcher", font=("Segoe UI Semibold", 16),
                 fg=TEXT, bg=BG).pack(side="left")
        tk.Label(hdr, text="CarRacing · Pong", font=FONT_UI,
                 fg=TEXT_DIM, bg=BG).pack(side="left", padx=(10, 0), pady=(3, 0))

        tk.Frame(self, height=1, bg=BORDER).pack(fill="x", padx=20, pady=(10, 0))

        # ── cards
        cards_frame = tk.Frame(self, bg=BG)
        cards_frame.pack(fill="x", padx=20, pady=14)
        cards_frame.columnconfigure(0, weight=1)
        cards_frame.columnconfigure(1, weight=1)

        # shared log widget (created before cards so cards can reference it)
        log_frame = tk.Frame(self, bg=BG)
        log_frame.pack(fill="both", expand=True, padx=20, pady=(0, 16))

        log_header = tk.Frame(log_frame, bg=BG)
        log_header.pack(fill="x", pady=(0, 4))
        tk.Label(log_header, text="Output", font=FONT_HEAD,
                 fg=TEXT_DIM, bg=BG).pack(side="left")
        tk.Button(log_header, text="Clear", font=FONT_UI,
                  fg=TEXT_DIM, bg=BG, activeforeground=TEXT,
                  activebackground=BG, relief="flat", bd=0,
                  cursor="hand2",
                  command=lambda: (
                      shared_log.configure(state="normal"),
                      shared_log.delete("1.0", tk.END),
                      shared_log.configure(state="disabled"),
                  )).pack(side="right")

        shared_log = scrolledtext.ScrolledText(
            log_frame, font=FONT_MONO, bg=PANEL, fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0,
            state="disabled", wrap="word",
            highlightthickness=1, highlightbackground=BORDER,
        )
        shared_log.pack(fill="both", expand=True)

        # CarRacing card
        car_card = Card(
            cards_frame, "CarRacing", ACCENT_CAR,
            actions=[
                {"label": "▶ Train", "key": "carracing_train",
                 "script": SCRIPTS["carracing_train"]},
                {"label": "▶ Play",  "key": "carracing_play",
                 "script": SCRIPTS["carracing_play"]},
            ],
            proc_mgr=pm, shared_log=shared_log,
        )
        car_card.grid(row=0, column=0, sticky="nsew", padx=(0, 7))

        # Pong card
        pong_card = Card(
            cards_frame, "Pong", ACCENT_PONG,
            actions=[
                {"label": "▶ Train",      "key": "pong_train",
                 "script": SCRIPTS["pong_train"]},
                {"label": "▶ Play",       "key": "pong_play",
                 "script": SCRIPTS["pong_play"]},
                {"label": "▶ Watch AI",   "key": "pong_watch",
                 "script": SCRIPTS["pong_watch"]},
            ],
            proc_mgr=pm, shared_log=shared_log,
        )
        pong_card.grid(row=0, column=1, sticky="nsew", padx=(7, 0))


if __name__ == "__main__":
    app = Launcher()
    app.mainloop()

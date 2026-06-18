"""
RL Launcher — GUI pentru CarRacing si Pong
"""
import tkinter as tk
from tkinter import ttk, scrolledtext
import subprocess
import threading
import sys
import os
import glob


BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
CARRACING_DIR = BASE_DIR
PONG_DIR     = os.path.join(BASE_DIR, "pong")

SCRIPTS = {
    "carracing_train": os.path.join(CARRACING_DIR, "train_gpu.py"),
    "carracing_play":  os.path.join(CARRACING_DIR, "test_trained.py"),
    "pong_train":      os.path.join(PONG_DIR, "train.py"),
    "pong_play":       os.path.join(PONG_DIR, "play.py"),
    "pong_watch":      os.path.join(PONG_DIR, "watch.py"),
}

BG          = "#0f0f13"
PANEL       = "#16161d"
BORDER      = "#2a2a38"
ACCENT_CAR  = "#e8a020"
ACCENT_PONG = "#3db8f5"
TEXT        = "#e0e0e8"
TEXT_DIM    = "#6b6b82"
GREEN       = "#4caf7d"
RED         = "#e05c5c"
FONT_MONO   = ("Consolas", 9)
FONT_UI     = ("Segoe UI", 9)
FONT_TITLE  = ("Segoe UI Semibold", 11)
FONT_HEAD   = ("Segoe UI Semibold", 9)


# ── helpers ───────────────────────────────────────────────────────────────────
def _append(widget, text):
    widget.configure(state="normal")
    widget.insert(tk.END, text)
    widget.see(tk.END)
    widget.configure(state="disabled")


def scan_models(folder, pattern="*.zip"):
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    return [os.path.basename(f) for f in files] or ["(no models found)"]


def make_dropdown(parent, var, values, width=28):
    cb = ttk.Combobox(parent, textvariable=var, values=values,
                      width=width, state="readonly", font=FONT_UI)
    if values:
        cb.current(0)
    return cb


def make_btn(parent, text, accent, command):
    b = tk.Button(parent, text=text, font=FONT_HEAD,
                  fg=BG, bg=accent, activeforeground=BG,
                  activebackground=accent, relief="flat", bd=0,
                  padx=12, pady=5, cursor="hand2", command=command)
    b.pack(side="left", padx=(0, 6))
    return b


# ── Process manager ───────────────────────────────────────────────────────────
class ProcessManager:
    def __init__(self):
        self._procs = {}

    def start(self, key, script, extra_args, log_widget, status_cb):
        if key in self._procs and self._procs[key].poll() is None:
            return
        def run():
            try:
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                proc = subprocess.Popen(
                    [sys.executable, "-u", script] + extra_args,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                    cwd=os.path.dirname(script), env=env,
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

    def stop(self, key):
        p = self._procs.get(key)
        if p and p.poll() is None:
            p.terminate()

    def is_running(self, key):
        p = self._procs.get(key)
        return p is not None and p.poll() is None


# ── CarRacing card ────────────────────────────────────────────────────────────
class CarRacingCard(tk.Frame):
    def __init__(self, parent, pm, log):
        super().__init__(parent, bg=PANEL, highlightthickness=1,
                         highlightbackground=BORDER)
        self.pm  = pm
        self.log = log
        self._status_color = TEXT_DIM
        self._build()

    def _build(self):
        tk.Frame(self, height=3, bg=ACCENT_CAR).pack(fill="x")
        inner = tk.Frame(self, bg=PANEL)
        inner.pack(fill="both", expand=True, padx=14, pady=10)

        # title + status
        row = tk.Frame(inner, bg=PANEL)
        row.pack(fill="x")
        tk.Label(row, text="CarRacing", font=FONT_TITLE,
                 fg=TEXT, bg=PANEL).pack(side="left")
        self._dot = tk.Canvas(row, width=10, height=10,
                              bg=PANEL, highlightthickness=0)
        self._dot_item = self._dot.create_oval(1,1,9,9, fill=TEXT_DIM, outline="")
        self._dot.pack(side="right")
        self._slbl = tk.Label(row, text="idle", font=FONT_UI,
                              fg=TEXT_DIM, bg=PANEL)
        self._slbl.pack(side="right", padx=(0,4))

        tk.Frame(inner, height=1, bg=BORDER).pack(fill="x", pady=(8,8))

        # Train section
        tk.Label(inner, text="TRAIN", font=FONT_HEAD,
                 fg=TEXT_DIM, bg=PANEL).pack(anchor="w")
        train_row = tk.Frame(inner, bg=PANEL)
        train_row.pack(fill="x", pady=(4,10))
        make_btn(train_row, "▶ Train", ACCENT_CAR,
                 lambda: self._launch("carracing_train", []))

        # Play section
        tk.Label(inner, text="PLAY", font=FONT_HEAD,
                 fg=TEXT_DIM, bg=PANEL).pack(anchor="w")

        model_row = tk.Frame(inner, bg=PANEL)
        model_row.pack(fill="x", pady=(4,0))
        tk.Label(model_row, text="Model:", font=FONT_UI,
                 fg=TEXT_DIM, bg=PANEL).pack(side="left", padx=(0,6))

        self._car_models = scan_models(os.path.join(CARRACING_DIR, "models"))
        self._car_model_var = tk.StringVar(value=self._car_models[0])

        self._car_dd = make_dropdown(model_row, self._car_model_var,
                                     self._car_models, width=26)
        self._car_dd.pack(side="left")

        tk.Button(model_row, text="↺", font=FONT_UI, fg=TEXT_DIM, bg=PANEL,
                  activeforeground=TEXT, activebackground=PANEL,
                  relief="flat", bd=0, cursor="hand2",
                  command=self._refresh_car_models).pack(side="left", padx=(4,0))

        play_row = tk.Frame(inner, bg=PANEL)
        play_row.pack(fill="x", pady=(6,0))
        make_btn(play_row, "▶ Play", ACCENT_CAR,
                 lambda: self._launch("carracing_play", [
                     "--model",
                     os.path.join(CARRACING_DIR, "models",
                                  self._car_model_var.get())
                 ]))

        # stop
        stop_row = tk.Frame(inner, bg=PANEL)
        stop_row.pack(fill="x", pady=(10,0))
        tk.Button(stop_row, text="■ Stop", font=FONT_HEAD,
                  fg=RED, bg=PANEL, activeforeground=RED,
                  activebackground=PANEL, relief="flat", bd=0,
                  padx=8, pady=5, cursor="hand2",
                  command=self._stop).pack(side="right")

    def _refresh_car_models(self):
        models = scan_models(os.path.join(CARRACING_DIR, "models"))
        self._car_dd["values"] = models
        if models:
            self._car_model_var.set(models[0])

    def _launch(self, key, args):
        script = SCRIPTS[key]
        if not os.path.exists(script):
            _append(self.log, f"[ERROR] Not found: {script}\n")
            return
        self._set_status("starting…", ACCENT_CAR)
        self.pm.start(key, script, args, self.log,
                      lambda s: self._set_status(
                          s, {GREEN:"running",RED:"error"}.get(
                              {"running":GREEN,"error":RED}.get(s, TEXT_DIM), TEXT_DIM)
                          if s != "running" else GREEN
                      ))

    def _stop(self):
        for k in ("carracing_train", "carracing_play"):
            self.pm.stop(k)
        self._set_status("stopped", TEXT_DIM)

    def _set_status(self, text, color):
        colors = {"running": GREEN, "stopped": TEXT_DIM,
                  "error": RED, "starting…": ACCENT_CAR}
        c = colors.get(text, color)
        self.after(0, lambda: (
            self._slbl.configure(text=text, fg=c),
            self._dot.itemconfig(self._dot_item, fill=c),
        ))


# ── Pong card ─────────────────────────────────────────────────────────────────
class PongCard(tk.Frame):
    def __init__(self, parent, pm, log):
        super().__init__(parent, bg=PANEL, highlightthickness=1,
                         highlightbackground=BORDER)
        self.pm  = pm
        self.log = log
        self._build()

    def _build(self):
        tk.Frame(self, height=3, bg=ACCENT_PONG).pack(fill="x")
        inner = tk.Frame(self, bg=PANEL)
        inner.pack(fill="both", expand=True, padx=14, pady=10)

        row = tk.Frame(inner, bg=PANEL)
        row.pack(fill="x")
        tk.Label(row, text="Pong", font=FONT_TITLE,
                 fg=TEXT, bg=PANEL).pack(side="left")
        self._dot = tk.Canvas(row, width=10, height=10,
                              bg=PANEL, highlightthickness=0)
        self._dot_item = self._dot.create_oval(1,1,9,9, fill=TEXT_DIM, outline="")
        self._dot.pack(side="right")
        self._slbl = tk.Label(row, text="idle", font=FONT_UI,
                              fg=TEXT_DIM, bg=PANEL)
        self._slbl.pack(side="right", padx=(0,4))

        tk.Frame(inner, height=1, bg=BORDER).pack(fill="x", pady=(8,8))

        pong_models = scan_models(os.path.join(PONG_DIR, "models"))

        # ── Train
        tk.Label(inner, text="TRAIN", font=FONT_HEAD,
                 fg=TEXT_DIM, bg=PANEL).pack(anchor="w")
        tr = tk.Frame(inner, bg=PANEL)
        tr.pack(fill="x", pady=(4,10))
        make_btn(tr, "▶ Train", ACCENT_PONG,
                 lambda: self._launch("pong_train", []))

        tk.Frame(inner, height=1, bg=BORDER).pack(fill="x", pady=(0,8))

        # ── Play (human)
        tk.Label(inner, text="PLAY vs HUMAN", font=FONT_HEAD,
                 fg=TEXT_DIM, bg=PANEL).pack(anchor="w")

        pm_row = tk.Frame(inner, bg=PANEL)
        pm_row.pack(fill="x", pady=(4,0))
        tk.Label(pm_row, text="Model:", font=FONT_UI,
                 fg=TEXT_DIM, bg=PANEL).pack(side="left", padx=(0,6))
        self._play_model_var = tk.StringVar(value=pong_models[0])
        self._play_dd = make_dropdown(pm_row, self._play_model_var,
                                      pong_models, width=24)
        self._play_dd.pack(side="left")
        tk.Button(pm_row, text="↺", font=FONT_UI, fg=TEXT_DIM, bg=PANEL,
                  activeforeground=TEXT, activebackground=PANEL,
                  relief="flat", bd=0, cursor="hand2",
                  command=lambda: self._refresh("play")).pack(side="left", padx=(4,0))

        pr = tk.Frame(inner, bg=PANEL)
        pr.pack(fill="x", pady=(6,10))
        make_btn(pr, "▶ Play", ACCENT_PONG,
                 lambda: self._launch("pong_play", [
                     "--model",
                     os.path.join(PONG_DIR, "models",
                                  self._play_model_var.get())
                 ]))

        tk.Frame(inner, height=1, bg=BORDER).pack(fill="x", pady=(0,8))

        # ── Watch AI
        tk.Label(inner, text="WATCH AI", font=FONT_HEAD,
                 fg=TEXT_DIM, bg=PANEL).pack(anchor="w")

        mode_row = tk.Frame(inner, bg=PANEL)
        mode_row.pack(fill="x", pady=(4,0))
        tk.Label(mode_row, text="Mode:", font=FONT_UI,
                 fg=TEXT_DIM, bg=PANEL).pack(side="left", padx=(0,6))
        self._watch_mode = tk.StringVar(value="single")
        ttk.Combobox(mode_row, textvariable=self._watch_mode,
                     values=["single", "double"], width=8,
                     state="readonly", font=FONT_UI).pack(side="left")

        # right model
        wr = tk.Frame(inner, bg=PANEL)
        wr.pack(fill="x", pady=(6,0))
        tk.Label(wr, text="Right:", font=FONT_UI,
                 fg=TEXT_DIM, bg=PANEL).pack(side="left", padx=(0,6))
        self._watch_right_var = tk.StringVar(value=pong_models[0])
        self._watch_right_dd = make_dropdown(wr, self._watch_right_var,
                                              pong_models, width=22)
        self._watch_right_dd.pack(side="left")
        tk.Button(wr, text="↺", font=FONT_UI, fg=TEXT_DIM, bg=PANEL,
                  activeforeground=TEXT, activebackground=PANEL,
                  relief="flat", bd=0, cursor="hand2",
                  command=lambda: self._refresh("watch")).pack(side="left", padx=(4,0))

        # left model (only for double)
        wl = tk.Frame(inner, bg=PANEL)
        wl.pack(fill="x", pady=(4,0))
        tk.Label(wl, text="Left: ", font=FONT_UI,
                 fg=TEXT_DIM, bg=PANEL).pack(side="left", padx=(0,6))
        self._watch_left_var = tk.StringVar(
            value=pong_models[-1] if len(pong_models) > 1 else pong_models[0])
        self._watch_left_dd = make_dropdown(wl, self._watch_left_var,
                                             pong_models, width=22)
        self._watch_left_dd.pack(side="left")

        wbtn = tk.Frame(inner, bg=PANEL)
        wbtn.pack(fill="x", pady=(6,10))
        make_btn(wbtn, "▶ Watch", ACCENT_PONG, self._launch_watch)

        # stop
        stop_row = tk.Frame(inner, bg=PANEL)
        stop_row.pack(fill="x")
        tk.Button(stop_row, text="■ Stop", font=FONT_HEAD,
                  fg=RED, bg=PANEL, activeforeground=RED,
                  activebackground=PANEL, relief="flat", bd=0,
                  padx=8, pady=5, cursor="hand2",
                  command=self._stop).pack(side="right")

    def _refresh(self, target):
        models = scan_models(os.path.join(PONG_DIR, "models"))
        if target == "play":
            self._play_dd["values"] = models
            if models: self._play_model_var.set(models[0])
        elif target == "watch":
            self._watch_right_dd["values"] = models
            self._watch_left_dd["values"]  = models
            if models:
                self._watch_right_var.set(models[0])
                self._watch_left_var.set(models[-1])

    def _launch_watch(self):
        mode = self._watch_mode.get()
        args = [
            "--mode",  mode,
            "--right", os.path.join(PONG_DIR, "models",
                                    self._watch_right_var.get()),
        ]
        if mode == "double":
            args += ["--left", os.path.join(PONG_DIR, "models",
                                             self._watch_left_var.get())]
        self._launch("pong_watch", args)

    def _launch(self, key, args):
        script = SCRIPTS[key]
        if not os.path.exists(script):
            _append(self.log, f"[ERROR] Not found: {script}\n")
            return
        self._set_status("starting…", ACCENT_PONG)
        self.pm.start(key, script, args, self.log,
                      lambda s: self._set_status(s, {
                          "running": GREEN, "error": RED
                      }.get(s, TEXT_DIM)))

    def _stop(self):
        for k in ("pong_train", "pong_play", "pong_watch"):
            self.pm.stop(k)
        self._set_status("stopped", TEXT_DIM)

    def _set_status(self, text, color):
        self.after(0, lambda: (
            self._slbl.configure(text=text, fg=color),
            self._dot.itemconfig(self._dot_item, fill=color),
        ))


# ── Main window ───────────────────────────────────────────────────────────────
class Launcher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RL Launcher")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(780, 640)
        self._build()

    def _build(self):
        pm = ProcessManager()

        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=20, pady=(16, 0))
        tk.Label(hdr, text="RL Launcher", font=("Segoe UI Semibold", 16),
                 fg=TEXT, bg=BG).pack(side="left")
        tk.Label(hdr, text="CarRacing · Pong", font=FONT_UI,
                 fg=TEXT_DIM, bg=BG).pack(side="left", padx=(10,0), pady=(3,0))
        tk.Frame(self, height=1, bg=BORDER).pack(fill="x", padx=20, pady=(10,0))

        # log
        log_frame = tk.Frame(self, bg=BG)
        log_frame.pack(side="bottom", fill="both", expand=True,
                       padx=20, pady=(0,16))
        lh = tk.Frame(log_frame, bg=BG)
        lh.pack(fill="x", pady=(0,4))
        tk.Label(lh, text="Output", font=FONT_HEAD,
                 fg=TEXT_DIM, bg=BG).pack(side="left")

        shared_log = scrolledtext.ScrolledText(
            log_frame, font=FONT_MONO, bg=PANEL, fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0,
            state="disabled", wrap="word",
            highlightthickness=1, highlightbackground=BORDER,
        )
        shared_log.pack(fill="both", expand=True)

        tk.Button(lh, text="Clear", font=FONT_UI, fg=TEXT_DIM, bg=BG,
                  activeforeground=TEXT, activebackground=BG,
                  relief="flat", bd=0, cursor="hand2",
                  command=lambda: (
                      shared_log.configure(state="normal"),
                      shared_log.delete("1.0", tk.END),
                      shared_log.configure(state="disabled"),
                  )).pack(side="right")

        # cards
        cards = tk.Frame(self, bg=BG)
        cards.pack(fill="x", padx=20, pady=14)
        cards.columnconfigure(0, weight=1)
        cards.columnconfigure(1, weight=1)

        CarRacingCard(cards, pm, shared_log).grid(
            row=0, column=0, sticky="nsew", padx=(0,7))
        PongCard(cards, pm, shared_log).grid(
            row=0, column=1, sticky="nsew", padx=(7,0))


if __name__ == "__main__":
    Launcher().mainloop()

#!/usr/bin/env python3
"""
codex_voice.py - Codex / GPT GUI with optional voice control (codex_voice branch).

- Builds on Version 1 (codex_gui.py) with model + reasoning dropdowns and Codex CLI integration.
- Adds microphone-based speech recognition using Vosk + sounddevice, all local.
- Adds text-to-speech using Coqui TTS (TTS library) + soundfile + sounddevice, all local.
- Wake word / assistant name is configurable via a text field (default: "Codex").
- Adds Guided vs Unguided mode:
    * Unguided (default): Only speaks when explicitly asked (e.g. "Codex respond" or Speak button).
    * Guided: Speaks short confirmations and help after commands or when it doesn't understand.

Voice commands (case-insensitive, recognized usually in lowercase):

  "<Name> stop listening"
      -> Turn off listening and stop the audio stream.

  "<Name> prompt"
      -> Enter "prompt capture" mode.
         All subsequent speech is appended to the Prompt field
         until you say "<Name> enter".

  "<Name> enter"
      -> Exit "prompt capture" mode. ("end" is accepted as a backup.)

  "<Name> model <name>"
      -> Set the model dropdown if <name> roughly matches a model
         in MODEL_OPTIONS.

  "<Name> model previous"
  "<Name> model next"
      -> Cycle backward/forward through the available models.

  "<Name> reasoning <none|low|medium|high>"
      -> Set the reasoning effort directly.

  "<Name> mode guided"
  "<Name> mode unguided"
      -> Switch between Guided and Unguided modes.

  "<Name> run"
      -> Trigger the Run action (same as clicking the button).

  "<Name> clear"
      -> Ask for confirmation before clearing prompt/response/console.
         While the confirm dialog is open, you can say
         "confirm" or "cancel" by voice.

  "<Name> respond"
      -> Read the current Response text aloud.

  "<Name> pause"
      -> Pause reading: stop after the current sentence and remember position.

  "<Name> continue"
      -> Resume reading from the next sentence.

  "<Name> stop"
      -> Stop reading aloud completely and clear TTS state.

  "<Name> help"
      -> Speak a short help message about voice commands.

Calibration:

  - Click "Calibrate Voice".
  - A window appears with lines like "<Name> mode guided", "<Name> mode unguided", etc.
  - Turn listening ON, click Calibrate, then read those lines with short pauses.
  - Click "Done" to:
      * learn additional wake-name variants,
      * learn extra "guided"/"unguided" variants,
      * open the Word Calibration table where you can review/edit aliases.

Per-word calibration:

  - Click "Word Calibration…", pick a word, then "Calibrate / Edit…".
  - In the dialog:
      * Edit aliases manually (one per line), OR
      * Turn on "Capture phrases from microphone while this window is open".
      * While that box is checked and Listening is ON, whatever Vosk hears
        is appended as a new line for that word.

Calibration profiles:

  - Auto-saved to: ~/.codex_voice_profile.json
  - In "Word Calibration…" you can:
      * Export Profile…  -> save JSON wherever you like
      * Import Profile…  -> load a JSON profile (replaces current aliases)
"""

import difflib
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np  # used to concatenate recording chunks

# --- Coqui TTS dependency (TTS + soundfile) -----------------------------------
try:
    from TTS.api import TTS as CoquiTTS
    import soundfile as sf

    COQUI_AVAILABLE = True
except ImportError:
    COQUI_AVAILABLE = False

# --- Voice recognition dependencies (Vosk + sounddevice) ----------------------
try:
    import sounddevice as sd
    from vosk import Model, KaldiRecognizer

    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

# Path to your local Vosk model directory.
VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"

# Coqui TTS model (multi-speaker, more natural voices)
COQUI_MODEL_NAME = "tts_models/en/vctk/vits"  # multi-speaker English model


# List of models to offer in the dropdown – customize if you like
MODEL_OPTIONS = [
    "gpt-5",
    "gpt-5-codex",
    "gpt-4.1",
    "gpt-4o",
]

REASONING_OPTIONS = [
    "none",
    "low",
    "medium",
    "high",
]

# ---------------------------------------------------------------------------
# Command vocabulary & per-word voice profile
# ---------------------------------------------------------------------------

COMMAND_PHRASES = [
    "{name} prompt",
    "{name} enter",
    "{name} run",
    "{name} respond",
    "{name} pause",
    "{name} continue",
    "{name} stop",
    "{name} stop listening",
    "{name} mode guided",
    "{name} mode unguided",
    "{name} model next",
    "{name} model previous",
    "{name} reasoning high",
    "{name} reasoning low",
    "{name} clear",
    "{name} help",
    # confirm/cancel for dialogs
    "confirm",
    "cancel",
    "yes",
    "no",
    "yeah",
    "hell yes",
]


class VoiceProfile:
    """
    Per-word calibration profile.

    Phase 1: we store *text aliases* for each logical word used in commands,
    plus metadata. These aliases are exactly what Vosk tends to output when
    you try to say that word.

    Data shape (in JSON):

    {
      "target_samples": 3,
      "words": {
        "guided": {
          "aliases": ["guy did", "guidance"],
          "last_calibrated": "2025-11-05T23:18:00Z",
          "status": "Calibrated"
        },
        ...
      }
    }
    """

    PROFILE_PATH = Path.home() / ".codex_voice_profile.json"

    def __init__(self, target_samples: int = 3):
        self.target_samples = target_samples
        # word -> {aliases: set[str], last_calibrated: str|None, status: str}
        self.words = {}

        # Build initial vocabulary
        self._init_vocab_from_commands()

        # Try to load persisted profile
        self.load()

    # --- Vocab ------------------------------------------------------------

    def _init_vocab_from_commands(self):
        base_words = set()
        for phrase in COMMAND_PHRASES:
            phrase_clean = phrase.replace("{name}", "").strip().lower()
            tokens = re.split(r"\s+", phrase_clean)
            for t in tokens:
                t = t.strip(" ,.!?'\"")
                if not t:
                    continue
                base_words.add(t)

        # Ensure yes/no/etc are present even if not in phrases
        base_words.update(["yes", "no", "yeah", "hell"])

        for w in sorted(base_words):
            if w not in self.words:
                self.words[w] = {
                    "aliases": set(),
                    "last_calibrated": None,
                    "status": "Uncalibrated",
                }

    def ensure_word(self, word: str):
        w = word.strip().lower()
        if not w:
            return
        if w not in self.words:
            self.words[w] = {
                "aliases": set(),
                "last_calibrated": None,
                "status": "Uncalibrated",
            }

    # --- Persistence ------------------------------------------------------

    def _serialize(self):
        out = {
            "target_samples": self.target_samples,
            "words": {},
        }
        for w, meta in self.words.items():
            out["words"][w] = {
                "aliases": sorted(meta["aliases"]),
                "last_calibrated": meta["last_calibrated"],
                "status": meta["status"],
            }
        return out

    def load(self):
        if not self.PROFILE_PATH.exists():
            return
        try:
            data = json.loads(self.PROFILE_PATH.read_text("utf-8"))
        except Exception:
            return

        target = data.get("target_samples")
        if isinstance(target, int) and target > 0:
            self.target_samples = target

        words_data = data.get("words", {})
        for w, meta in words_data.items():
            if w not in self.words:
                self.words[w] = {
                    "aliases": set(),
                    "last_calibrated": None,
                    "status": "Uncalibrated",
                }
            aliases = set(meta.get("aliases", []))
            self.words[w]["aliases"] = aliases
            self.words[w]["last_calibrated"] = meta.get("last_calibrated")
            self.words[w]["status"] = meta.get("status", "Uncalibrated")

    def save(self):
        try:
            self.PROFILE_PATH.write_text(
                json.dumps(self._serialize(), indent=2), encoding="utf-8"
            )
        except Exception:
            pass

    # external save/load for explicit calibration files
    def export_to_file(self, path: str):
        try:
            Path(path).write_text(
                json.dumps(self._serialize(), indent=2),
                encoding="utf-8",
            )
        except Exception:
            raise

    def import_from_file(self, path: str, replace: bool = True):
        data = json.loads(Path(path).read_text("utf-8"))

        target = data.get("target_samples")
        if isinstance(target, int) and target > 0:
            self.target_samples = target

        words_data = data.get("words", {})
        if replace:
            for w in list(self.words.keys()):
                self.words[w]["aliases"].clear()
                self.words[w]["last_calibrated"] = None
                self.words[w]["status"] = "Uncalibrated"

        for w, meta in words_data.items():
            if w not in self.words:
                self.words[w] = {
                    "aliases": set(),
                    "last_calibrated": None,
                    "status": "Uncalibrated",
                }
            aliases = set(meta.get("aliases", []))
            self.words[w]["aliases"] = aliases
            self.words[w]["last_calibrated"] = meta.get("last_calibrated")
            self.words[w]["status"] = meta.get("status", "Uncalibrated")

        # recompute statuses
        for w in self.words:
            self.words[w]["status"] = self._compute_status_for_word(w)

    # --- Status computation -----------------------------------------------

    def _compute_status_for_word(self, word: str):
        meta = self.words[word]
        n = len(meta["aliases"])
        if n == 0:
            return "Uncalibrated"
        if n < self.target_samples:
            return "Partial"
        return meta.get("status") or "Calibrated"

    def get_status(self, word: str):
        meta = self.words[word]
        meta["status"] = self._compute_status_for_word(word)
        return meta["status"]

    def set_target_samples(self, target: int):
        if target <= 0:
            return
        self.target_samples = target
        for w in self.words:
            self.words[w]["status"] = self._compute_status_for_word(w)

    # --- Aliases ----------------------------------------------------------

    def get_aliases(self, word: str):
        word = word.lower()
        if word not in self.words:
            return []
        return sorted(self.words[word]["aliases"])

    def set_aliases(self, word: str, aliases):
        word = word.lower()
        self.ensure_word(word)
        aliases_clean = set()
        for a in aliases:
            a = a.strip()
            if not a:
                continue
            aliases_clean.add(a.lower())
        self.words[word]["aliases"] = aliases_clean
        self.words[word]["last_calibrated"] = datetime.utcnow().isoformat() + "Z"
        self.words[word]["status"] = self._compute_status_for_word(word)

    def add_alias(self, word: str, alias: str):
        word = word.lower()
        alias = alias.strip()
        if not alias:
            return
        self.ensure_word(word)
        self.words[word]["aliases"].add(alias.lower())
        self.words[word]["last_calibrated"] = datetime.utcnow().isoformat() + "Z"
        self.words[word]["status"] = self._compute_status_for_word(word)

    def delete_aliases(self, word: str, aliases=None):
        word = word.lower()
        if word not in self.words:
            return
        if aliases is None or aliases == "all":
            self.words[word]["aliases"].clear()
        else:
            for a in aliases:
                self.words[word]["aliases"].discard(a.lower())
        self.words[word]["status"] = self._compute_status_for_word(word)


class CodexVoiceGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Codex / GPT Voice Frontend")
        self.geometry("1200x800")

        # Base directories
        self.base_dir = Path(__file__).resolve().parent
        self.recordings_dir = self.base_dir / "recordings"
        self.log_dir = self.base_dir / "logs"

        # Voice-recognition-related state
        # voice_mode: 'idle', 'prompt', 'confirm_clear', 'calibrate', 'word_calib:<word>'
        self.listening = False
        self.voice_mode = "idle"
        self.audio_queue = queue.Queue()
        self.vosk_model = None
        self.vosk_recognizer = None
        self.voice_thread = None
        self.audio_stream = None
        self.sample_rate = 16000
        self.clear_confirm_window = None

        # Calibration state
        self.calibration_window = None
        self.calib_heard_text = None
        self.calibration_captured = []

        # Coqui TTS-related state
        self.coqui_tts = None
        self.coqui_speakers = []
        self.voice_display_map = {}

        # TTS playback state
        self.tts_thread = None
        self.tts_chunks = []
        self.tts_index = 0
        self.tts_paused = False
        self.tts_should_stop = False
        self.tts_lock = threading.Lock()
        self.tts_last_stop = 0.0
        self.tts_recording_path = None  # where to save combined TTS recording for responses

        # Per-word voice profile
        self.voice_profile = VoiceProfile(target_samples=3)

        # Word calibration UI state (table)
        self.word_calib_window = None
        self.word_filter_var = None
        self.word_status_filter_var = None
        self.word_tree = None
        self.target_samples_var = None

        # Individual word edit dialog state
        self.word_edit_window = None
        self.word_edit_text = None
        self.word_edit_word = None
        self.word_edit_info_var = None
        self.word_edit_voice_enabled = None

        # Logging state
        self.debug_log_file = None
        self.pr_log_file = None

        self._build_widgets()

        # Global exception hook -> debug log when enabled
        sys.excepthook = self._handle_exception

    # -------------------------------------------------------------------------
    # UI setup
    # -------------------------------------------------------------------------
    def _build_widgets(self):
        control_frame = ttk.Frame(self)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Model selection
        ttk.Label(control_frame, text="Model:").grid(row=0, column=0, sticky="w")
        self.model_var = tk.StringVar(value=MODEL_OPTIONS[0])
        self.model_menu = ttk.Combobox(
            control_frame,
            textvariable=self.model_var,
            values=MODEL_OPTIONS,
            state="readonly",
            width=18,
        )
        self.model_menu.grid(row=0, column=1, padx=(5, 15), sticky="w")

        # Reasoning
        ttk.Label(control_frame, text="Reasoning Effort:").grid(row=0, column=2, sticky="w")
        self.reasoning_var = tk.StringVar(value="medium")
        self.reasoning_menu = ttk.Combobox(
            control_frame,
            textvariable=self.reasoning_var,
            values=REASONING_OPTIONS,
            state="readonly",
            width=10,
        )
        self.reasoning_menu.grid(row=0, column=3, padx=(5, 15), sticky="w")

        # Run
        self.run_button = ttk.Button(control_frame, text="Run", command=self.on_run)
        self.run_button.grid(row=0, column=4, padx=(5, 10))

        # Listening toggle
        self.listen_button = ttk.Button(
            control_frame,
            text="🎤 Listening: OFF",
            command=self.toggle_listening,
        )
        self.listen_button.grid(row=0, column=5, padx=(5, 15))

        # Assistant name
        ttk.Label(control_frame, text="Assistant name:").grid(row=0, column=6, sticky="w")
        self.wake_name_var = tk.StringVar(value="Codex")
        self.wake_entry = ttk.Entry(control_frame, textvariable=self.wake_name_var, width=12)
        self.wake_entry.grid(row=0, column=7, padx=(5, 0), sticky="w")

        # Voice
        ttk.Label(control_frame, text="Voice:").grid(row=0, column=8, sticky="w")
        self.voice_var = tk.StringVar(value="Default")
        self.voice_menu = ttk.Combobox(
            control_frame,
            textvariable=self.voice_var,
            state="readonly",
            width=22,
        )
        self.voice_menu.grid(row=0, column=9, padx=(5, 0), sticky="w")

        # Speak Response
        self.speak_button = ttk.Button(
            control_frame,
            text="Speak Response",
            command=self._start_tts_read_response,
        )
        self.speak_button.grid(row=0, column=10, padx=(5, 0), sticky="w")

        # Status
        self.status_var = tk.StringVar(value="Ready.")
        self.status_label = ttk.Label(control_frame, textvariable=self.status_var)
        self.status_label.grid(row=1, column=0, columnspan=11, sticky="w", pady=(8, 0))

        # Mode
        ttk.Label(control_frame, text="Mode:").grid(row=2, column=0, sticky="w", pady=(4, 0))
        self.mode_var = tk.StringVar(value="Unguided")
        self.mode_menu = ttk.Combobox(
            control_frame,
            textvariable=self.mode_var,
            values=["Unguided", "Guided"],
            state="readonly",
            width=10,
        )
        self.mode_menu.grid(row=2, column=1, padx=(5, 0), sticky="w", pady=(4, 0))

        # Calibration buttons
        self.calib_button = ttk.Button(
            control_frame,
            text="Calibrate Voice",
            command=self._open_calibration,
        )
        self.calib_button.grid(row=2, column=2, padx=(5, 0), sticky="w", pady=(4, 0))

        self.word_calib_button = ttk.Button(
            control_frame,
            text="Word Calibration…",
            command=self._open_word_calibration,
        )
        self.word_calib_button.grid(row=2, column=3, padx=(5, 0), sticky="w", pady=(4, 0))

        # Logging / recording options
        self.save_recordings_var = tk.BooleanVar(value=True)
        self.debug_logging_var = tk.BooleanVar(value=False)
        self.log_prompts_var = tk.BooleanVar(value=False)

        ttk.Checkbutton(
            control_frame,
            text="Save recordings",
            variable=self.save_recordings_var,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))

        ttk.Checkbutton(
            control_frame,
            text="Debug logging",
            variable=self.debug_logging_var,
            command=self._on_debug_logging_toggle,
        ).grid(row=3, column=2, columnspan=2, sticky="w", pady=(4, 0))

        ttk.Checkbutton(
            control_frame,
            text="Log prompts/responses",
            variable=self.log_prompts_var,
            command=self._on_pr_logging_toggle,
        ).grid(row=3, column=4, columnspan=3, sticky="w", pady=(4, 0))

        # Main panes
        paned = ttk.PanedWindow(self, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Prompt
        prompt_frame = ttk.LabelFrame(paned, text="Prompt")
        self.prompt_text = tk.Text(prompt_frame, wrap="word", height=8)
        prompt_scroll = ttk.Scrollbar(
            prompt_frame, orient="vertical", command=self.prompt_text.yview
        )
        self.prompt_text.configure(yscrollcommand=prompt_scroll.set)
        self.prompt_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        prompt_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Response (editable)
        output_frame = ttk.LabelFrame(paned, text="Response")
        self.output_text = tk.Text(output_frame, wrap="word", height=12, state="normal")
        output_scroll = ttk.Scrollbar(
            output_frame, orient="vertical", command=self.output_text.yview
        )
        self.output_text.configure(yscrollcommand=output_scroll.set)
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        output_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Console
        console_frame = ttk.LabelFrame(paned, text="Console (Codex CLI + Voice log)")
        self.console_text = tk.Text(
            console_frame,
            wrap="word",
            height=8,
            state="normal",
            bg="#111111",
            fg="#eeeeee",
        )
        console_scroll = ttk.Scrollbar(
            console_frame, orient="vertical", command=self.console_text.yview
        )
        self.console_text.configure(yscrollcommand=console_scroll.set)
        self.console_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        console_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        paned.add(prompt_frame, weight=1)
        paned.add(output_frame, weight=2)
        paned.add(console_frame, weight=1)

        # Init TTS
        if COQUI_AVAILABLE and self._init_tts():
            self._populate_voice_dropdown()
        else:
            self.voice_menu["values"] = ["(TTS not available)"]
            self.voice_var.set("(TTS not available)")
            self.voice_menu.config(state="disabled")
            self.speak_button.config(state="disabled")

        self.bind_all("<Control-Return>", lambda event: self.on_run())
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # -------------------------------------------------------------------------
    # Logging helpers
    # -------------------------------------------------------------------------
    def _on_debug_logging_toggle(self, *args):
        if not self.debug_logging_var.get():
            if self.debug_log_file:
                try:
                    self.debug_log_file.close()
                except Exception:
                    pass
                self.debug_log_file = None
        else:
            self._debug_log("=== Debug logging enabled ===")

    def _on_pr_logging_toggle(self, *args):
        if not self.log_prompts_var.get():
            if self.pr_log_file:
                try:
                    self.pr_log_file.close()
                except Exception:
                    pass
                self.pr_log_file = None
        else:
            self._pr_log("=== Prompt/response logging enabled ===")

    def _ensure_debug_log_file(self):
        if not self.debug_logging_var.get():
            return None
        if self.debug_log_file is None:
            try:
                self.log_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            path = self.log_dir / f"debug_{ts}.log"
            try:
                f = open(path, "a", encoding="utf-8")
                f.write(f"# Debug log started at {datetime.now().isoformat()}\n")
                f.flush()
                self.debug_log_file = f
            except Exception:
                self.debug_log_file = None
                return None
        return self.debug_log_file

    def _debug_log(self, text: str):
        f = self._ensure_debug_log_file()
        if not f:
            return
        try:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
            f.flush()
        except Exception:
            pass

    def _ensure_pr_log_file(self):
        if not self.log_prompts_var.get():
            return None
        if self.pr_log_file is None:
            try:
                self.log_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            path = self.log_dir / f"prompts_{ts}.log"
            try:
                f = open(path, "a", encoding="utf-8")
                f.write(f"# Prompt/response log started at {datetime.now().isoformat()}\n")
                f.flush()
                self.pr_log_file = f
            except Exception:
                self.pr_log_file = None
                return None
        return self.pr_log_file

    def _pr_log(self, text: str):
        f = self._ensure_pr_log_file()
        if not f:
            return
        try:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
            f.flush()
        except Exception:
            pass

    def _handle_exception(self, exc_type, exc_value, exc_tb):
        import traceback as tb

        msg = "[CRASH] Unhandled exception:\n" + "".join(
            tb.format_exception(exc_type, exc_value, exc_tb)
        )
        try:
            self._debug_log(msg)
        except Exception:
            pass
        try:
            self._append_console(msg + "\n")
        except Exception:
            pass
        # still let default handler print to stderr
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    # -------------------------------------------------------------------------
    # Wake name & fuzzy helpers
    # -------------------------------------------------------------------------
    def _get_wake_name_lower(self) -> str:
        name = self.wake_name_var.get().strip()
        if not name:
            name = "Codex"
        self.voice_profile.ensure_word(name.lower())
        return name.lower()

    def _wake_targets_lower(self):
        base = self._get_wake_name_lower()
        aliases = self.voice_profile.get_aliases(base)
        targets = [base] + aliases
        if base == "codex":
            targets.extend(
                ["kodak", "kodaks", "kodak's", "kodack", "kodex", "kodax"]
            )
        return list(dict.fromkeys(t.lower() for t in targets))

    def _fuzzy(self, word: str, targets, threshold: float = 0.78) -> bool:
        if not word:
            return False
        word_norm = str(word).lower().strip(" ,.!?'\"")
        if not word_norm:
            return False

        best = 0.0
        for t in targets:
            t_norm = str(t).lower().strip(" ,.!?'\"")
            if not t_norm:
                continue
            ratio = difflib.SequenceMatcher(None, word_norm, t_norm).ratio()
            if ratio > best:
                best = ratio
        return best >= threshold

    def _is_guided(self) -> bool:
        try:
            return self.mode_var.get().lower().startswith("guided")
        except Exception:
            return False

    def _guided_say(self, text: str):
        if not text or not self._is_guided():
            return
        if not COQUI_AVAILABLE:
            return
        if self.tts_thread is not None and self.tts_thread.is_alive():
            return
        if self.tts_paused:
            return
        self._start_tts_from_text(text)

    # -------------------------------------------------------------------------
    # Coqui TTS helpers
    # -------------------------------------------------------------------------
    def _init_tts(self) -> bool:
        if not COQUI_AVAILABLE:
            self._append_console("[VOICE] Coqui TTS not available (install TTS + soundfile).\n")
            return False

        if self.coqui_tts is not None:
            return True

        try:
            self._append_console(
                f"[VOICE] Loading Coqui TTS model '{COQUI_MODEL_NAME}' "
                "(first run may take a while)...\n"
            )
            self.coqui_tts = CoquiTTS(
                model_name=COQUI_MODEL_NAME,
                progress_bar=False,
                gpu=False,
            )
            speakers = getattr(self.coqui_tts, "speakers", None)
            self.coqui_speakers = list(speakers) if speakers is not None else []
        except Exception as e:
            self._append_console(f"[VOICE] Could not initialize Coqui TTS: {e}\n")
            self.coqui_tts = None
            self.coqui_speakers = []
            return False

        return True

    def _populate_voice_dropdown(self):
        display_names = []
        self.voice_display_map = {}
        if self.coqui_speakers:
            limit = min(8, len(self.coqui_speakers))
            for idx in range(limit):
                spk = self.coqui_speakers[idx]
                disp = f"Voice {idx + 1} ({spk})"
                display_names.append(disp)
                self.voice_display_map[disp] = spk
        else:
            display_names = ["Default Voice"]
            self.voice_display_map["Default Voice"] = None

        self.voice_menu["values"] = display_names
        self.voice_var.set(display_names[0])

    def _get_selected_speaker_id(self):
        if not self.voice_display_map:
            return None
        sel = self.voice_var.get().strip()
        if not sel:
            return None
        return self.voice_display_map.get(sel)

    def _chunk_text_for_tts(self, text: str, max_chars: int = 300):
        text = text.strip()
        if not text:
            return []
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current = ""
        for s in sentences:
            if not s:
                continue
            if not current:
                current = s
            elif len(current) + 1 + len(s) <= max_chars:
                current += " " + s
            else:
                chunks.append(current)
                current = s
        if current:
            chunks.append(current)
        return chunks

    # === Core TTS control ====================================================
    def _tts_loop(self):
        speaker_id = self._get_selected_speaker_id()
        rec_path = self.tts_recording_path
        recording_chunks = []
        recording_sr = None

        try:
            while self.tts_index < len(self.tts_chunks) and not self.tts_should_stop:
                chunk = self.tts_chunks[self.tts_index]
                try:
                    with self.tts_lock:
                        if self.tts_should_stop:
                            break

                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                        tmp_path = tmp.name
                        tmp.close()

                        kwargs = {"text": chunk, "file_path": tmp_path}
                        if speaker_id:
                            kwargs["speaker"] = speaker_id

                        self.coqui_tts.tts_to_file(**kwargs)

                        if self.tts_should_stop:
                            try:
                                os.remove(tmp_path)
                            except OSError:
                                pass
                            break

                        data, sr = sf.read(tmp_path, dtype="float32")
                        if self.tts_should_stop:
                            try:
                                os.remove(tmp_path)
                            except OSError:
                                pass
                            break

                        # Append to in-memory recording buffer if requested
                        if rec_path is not None:
                            if recording_sr is None:
                                recording_sr = sr
                            recording_chunks.append(data.copy())

                        sd.play(data, sr)
                        sd.wait()

                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass

                except Exception as e:
                    self.after(
                        0,
                        self._append_console,
                        f"[VOICE] TTS error: {e}\n",
                    )
                    break

                self.tts_index += 1

        finally:
            # Save combined recording if requested
            if rec_path is not None and recording_chunks and recording_sr is not None:
                try:
                    self.recordings_dir.mkdir(parents=True, exist_ok=True)
                    full = np.concatenate(recording_chunks, axis=0)
                    sf.write(str(rec_path), full, recording_sr)
                    self._append_console(f"[VOICE] Saved recording to {rec_path}\n")
                except Exception as e:
                    self._append_console(f"[VOICE] Failed to save recording: {e}\n")

            self.tts_recording_path = None

            if self.tts_index >= len(self.tts_chunks):
                self.tts_paused = False
            self.tts_thread = None

    def _start_tts_from_text(self, text: str, record_label: str = None):
        text = text.strip()
        if not text:
            self._append_console("[VOICE] Nothing to read aloud.\n")
            return
        if not self._init_tts():
            return

        if self.tts_thread is not None and self.tts_thread.is_alive():
            now = time.time()
            if self.tts_last_stop and now - self.tts_last_stop < 1.5:
                self._append_console(
                    "[VOICE] Previous speech is still winding down; "
                    "wait a moment and try again.\n"
                )
            else:
                self._append_console(
                    "[VOICE] Speech already in progress; say stop, "
                    "then wait a moment before starting again.\n"
                )
            return

        self.tts_chunks = self._chunk_text_for_tts(text, max_chars=320)
        self.tts_index = 0
        self.tts_should_stop = False
        self.tts_paused = False

        # decide recording path (only for responses and if enabled)
        if record_label and self.save_recordings_var.get():
            try:
                ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                safe_label = re.sub(r"[^a-zA-Z0-9_-]+", "_", record_label)
                filename = f"{safe_label}_{ts}.wav"
                self.tts_recording_path = self.recordings_dir / filename
            except Exception:
                self.tts_recording_path = None
        else:
            self.tts_recording_path = None

        if not self.tts_chunks:
            self._append_console("[VOICE] Nothing to speak after cleaning.\n")
            return

        self.tts_thread = threading.Thread(target=self._tts_loop, daemon=True)
        self.tts_thread.start()
        self._append_console("[VOICE] Speaking via Coqui.\n")

    def _clean_response_text(self, text: str) -> str:
        lines = text.splitlines()
        kept = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            low = stripped.lower()
            if low.startswith("model:") or low.startswith("reasoning:"):
                continue
            kept.append(stripped)

        if not kept:
            return ""

        joined = " ".join(kept)
        joined = re.sub(r"[*_`#>]+", "", joined)
        joined = re.sub(r"\s*[-•]\s+", " ", joined)
        joined = re.sub(r"\s+", " ", joined)
        return joined.strip()

    def _start_tts_read_response(self):
        raw = self.output_text.get("1.0", tk.END).strip()
        if not raw:
            self._append_console("[VOICE] No response to read aloud.\n")
            return
        cleaned = self._clean_response_text(raw)
        if not cleaned:
            self._append_console(
                "[VOICE] Response contained only headers/markup; nothing to read.\n"
            )
            return

        # For response reading, record if enabled
        record_label = "response" if self.save_recordings_var.get() else None
        self._start_tts_from_text(cleaned, record_label=record_label)

    def _pause_tts(self):
        if self.tts_thread is None or not self.tts_thread.is_alive():
            self._append_console("[VOICE] No active speech to pause.\n")
            return
        if self.tts_paused:
            self._append_console("[VOICE] Speech is already paused.\n")
            return

        self.tts_should_stop = True
        self.tts_paused = True
        self._append_console("[VOICE] Pausing speech (after current sentence).\n")

    def _continue_tts(self):
        if not self.tts_paused:
            self._append_console("[VOICE] Speech is not paused.\n")
            return
        if not self.tts_chunks:
            self._append_console("[VOICE] Nothing to continue.\n")
            self.tts_paused = False
            return
        if self.tts_thread is not None and self.tts_thread.is_alive():
            self._append_console("[VOICE] Still finishing previous pause; try again shortly.\n")
            return
        if self.tts_index >= len(self.tts_chunks):
            self._append_console("[VOICE] Already at end of speech.\n")
            self.tts_paused = False
            return

        self.tts_should_stop = False
        self.tts_paused = False
        self.tts_thread = threading.Thread(target=self._tts_loop, daemon=True)
        self.tts_thread.start()
        self._append_console("[VOICE] Resuming speech via Coqui.\n")

    def _stop_tts_command(self):
        if self.tts_thread is None or not self.tts_thread.is_alive():
            self.tts_chunks = []
            self.tts_index = 0
            self.tts_paused = False
            self.tts_should_stop = False
            self._append_console("[VOICE] No active speech to stop.\n")
            return

        self.tts_should_stop = True
        self.tts_paused = False
        self.tts_last_stop = time.time()
        self._append_console("[VOICE] Stopping speech.\n")

    def _speak_help(self):
        name_raw = self.wake_name_var.get().strip() or "Codex"
        name = name_raw
        help_text = (
            f"{name} voice help. "
            f"Say '{name} prompt' to dictate a prompt, then speak your text. "
            f"Say '{name} enter' to stop dictation. "
            f"Say '{name} run' to submit the prompt. "
            f"Say '{name} respond' to read the latest response aloud. "
            f"Say '{name} pause', '{name} continue', or '{name} stop' to control speech playback. "
            f"Say '{name} mode guided' or '{name} mode unguided' to control how talkative I am. "
            f"Say '{name} model next' or '{name} model previous' to change models. "
            f"Say '{name} reasoning high' or '{name} reasoning low' to change reasoning effort. "
            f"And say '{name} stop listening' to turn off the microphone."
        )
        self._start_tts_from_text(help_text)

    # -------------------------------------------------------------------------
    # Calibration UI (global)
    # -------------------------------------------------------------------------
    def _open_calibration(self):
        if not VOSK_AVAILABLE:
            messagebox.showerror(
                "Voice unavailable",
                "Vosk and/or sounddevice not installed.\n\n"
                "Install with:\n"
                "  pip install vosk sounddevice\n"
                "and download a Vosk model, then update VOSK_MODEL_PATH.",
            )
            return

        if not self.listening:
            messagebox.showinfo(
                "Listening required",
                "Turn listening ON before calibration so your mic is active.",
            )
            return

        if self.voice_mode == "calibrate":
            return

        self.calibration_captured = []
        self.voice_mode = "calibrate"

        win = tk.Toplevel(self)
        win.title("Voice Calibration")
        self.calibration_window = win

        name_raw = self.wake_name_var.get().strip() or "Codex"
        script_lines = [
            f"{name_raw} mode guided",
            f"{name_raw} mode unguided",
            f"{name_raw} prompt",
            f"{name_raw} reasoning high",
            f"{name_raw} reasoning low",
        ]

        label = ttk.Label(
            win,
            text="Read each line out loud, with a short pause between:",
            justify="left",
        )
        label.pack(padx=10, pady=(10, 4), anchor="w")

        script_box = tk.Text(win, height=len(script_lines) + 1, width=40)
        script_box.insert("1.0", "\n".join(script_lines))
        script_box.config(state="disabled")
        script_box.pack(padx=10, pady=(0, 8), fill="x")

        ttk.Label(win, text="Heard phrases:").pack(padx=10, anchor="w")
        self.calib_heard_text = tk.Text(win, height=8, width=60, state="disabled")
        self.calib_heard_text.pack(padx=10, pady=(0, 8), fill="both", expand=True)

        btn_frame = ttk.Frame(win)
        btn_frame.pack(pady=(0, 10))
        ttk.Button(btn_frame, text="Done", command=self._finish_calibration).pack(
            side="left", padx=5
        )
        ttk.Button(btn_frame, text="Cancel", command=self._cancel_calibration).pack(
            side="left", padx=5
        )

        win.protocol("WM_DELETE_WINDOW", self._cancel_calibration)
        self._center_window(win, 500, 380)

        self._append_console(
            "[VOICE] Calibration started. Read the lines in the calibration window.\n"
        )
        self._guided_say(
            "Calibration started. Please read the lines in the calibration window, then click Done."
        )

    def _append_calibration_heard(self, text: str):
        if self.calib_heard_text is None:
            return
        self.calib_heard_text.config(state="normal")
        self.calib_heard_text.insert(tk.END, text + "\n")
        self.calib_heard_text.see(tk.END)
        self.calib_heard_text.config(state="disabled")

    def _apply_calibration_samples(self):
        if not self.calibration_captured:
            self._append_console("[VOICE] Calibration finished with no samples.\n")
            return

        base_name = self._get_wake_name_lower()
        mode_heads = ["mode", "node", "note", "mowed", "mold", "modee"]

        for idx, phrase in enumerate(self.calibration_captured):
            phrase = phrase.strip()
            if not phrase:
                continue
            tokens = phrase.split()
            if not tokens:
                continue

            first = tokens[0].strip(",.!?")
            if first and first != base_name:
                self.voice_profile.add_alias(base_name, first)

            if idx in (0, 1):
                j_mode = None
                for j, t in enumerate(tokens):
                    if self._fuzzy(t, mode_heads, threshold=0.70):
                        j_mode = j
                        break
                if j_mode is None or j_mode + 1 >= len(tokens):
                    continue

                tail = [
                    t.strip(",.!?").lower()
                    for t in tokens[j_mode + 1 :]
                    if t.strip(",.!?")
                ]

                if idx == 0:
                    for t in tail:
                        if "guid" in t or self._fuzzy(t, ["guided"], threshold=0.8):
                            self.voice_profile.add_alias("guided", t)
                else:
                    for t in tail:
                        if self._fuzzy(t, ["guided"], threshold=0.9):
                            continue
                        if "unguid" in t:
                            self.voice_profile.add_alias("unguided", t)
                        elif t.startswith("un") or t.startswith("on"):
                            self.voice_profile.add_alias("unguided", t)

        self.voice_profile.save()

        guided_aliases = self.voice_profile.get_aliases("guided")
        unguided_aliases = self.voice_profile.get_aliases("unguided")
        wake_aliases = self.voice_profile.get_aliases(base_name)

        if wake_aliases:
            self._append_console(
                f"[VOICE] Calibration added wake variants: {sorted(wake_aliases)}\n"
            )
        if guided_aliases:
            self._append_console(
                f"[VOICE] Calibration added guided variants: {sorted(guided_aliases)}\n"
            )
        if unguided_aliases:
            self._append_console(
                f"[VOICE] Calibration added unguided variants: {sorted(unguided_aliases)}\n"
            )

    def _finish_calibration(self):
        self._apply_calibration_samples()
        self.voice_mode = "idle"
        if self.calibration_window is not None:
            try:
                self.calibration_window.destroy()
            except Exception:
                pass
            self.calibration_window = None
            self.calib_heard_text = None

        self._open_word_calibration()
        self._guided_say("Calibration finished.")

    def _cancel_calibration(self):
        self._append_console("[VOICE] Calibration cancelled.\n")
        self.voice_mode = "idle"
        if self.calibration_window is not None:
            try:
                self.calibration_window.destroy()
            except Exception:
                pass
            self.calibration_window = None
            self.calib_heard_text = None

    # -------------------------------------------------------------------------
    # Per-word calibration table UI
    # -------------------------------------------------------------------------
    def _open_word_calibration(self):
        if self.word_calib_window is not None and self.word_calib_window.winfo_exists():
            self.word_calib_window.lift()
            return

        win = tk.Toplevel(self)
        win.title("Per-Word Voice Calibration")
        self.word_calib_window = win

        desc = ttk.Label(
            win,
            text=(
                "Calibrate how your assistant hears each command word.\n"
                "Each row is a word used in voice commands. Aliases are the\n"
                "actual strings Vosk tends to output when you say that word.\n"
                "You can edit aliases, recalibrate, or clear them.\n\n"
                "Use Export Profile… / Import Profile… to save/load calibration files."
            ),
            justify="left",
            wraplength=640,
        )
        desc.pack(padx=10, pady=(10, 8), anchor="w")

        top_frame = ttk.Frame(win)
        top_frame.pack(fill="x", padx=10, pady=(0, 6))

        ttk.Label(top_frame, text="Filter:").pack(side="left")
        self.word_filter_var = tk.StringVar()
        filter_entry = ttk.Entry(top_frame, textvariable=self.word_filter_var, width=20)
        filter_entry.pack(side="left", padx=(4, 8))

        ttk.Label(top_frame, text="Status:").pack(side="left")
        self.word_status_filter_var = tk.StringVar(value="All")
        status_combo = ttk.Combobox(
            top_frame,
            textvariable=self.word_status_filter_var,
            state="readonly",
            width=12,
            values=["All", "Uncalibrated", "Partial", "Calibrated", "Needs Review"],
        )
        status_combo.pack(side="left", padx=(4, 12))

        ttk.Label(top_frame, text="Target samples per word:").pack(side="left")
        self.target_samples_var = tk.IntVar(value=self.voice_profile.target_samples)
        target_spin = ttk.Spinbox(
            top_frame,
            from_=1,
            to=10,
            textvariable=self.target_samples_var,
            width=4,
            command=self._update_target_samples_from_ui,
        )
        target_spin.pack(side="left", padx=(4, 8))

        ttk.Button(top_frame, text="Calibrate All", command=self._calibrate_all_words).pack(
            side="left", padx=(10, 0)
        )

        columns = ("word", "status", "samples", "last_calib")
        tree = ttk.Treeview(
            win,
            columns=columns,
            show="headings",
            selectmode="browse",
            height=16,
        )
        self.word_tree = tree

        tree.heading("word", text="Word")
        tree.heading("status", text="Status")
        tree.heading("samples", text="Samples")
        tree.heading("last_calib", text="Last Calibrated")

        tree.column("word", width=120, anchor="w")
        tree.column("status", width=110, anchor="center")
        tree.column("samples", width=90, anchor="center")
        tree.column("last_calib", width=200, anchor="w")

        tree.pack(fill="both", expand=True, padx=10, pady=(0, 6))

        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill="x", padx=10, pady=(0, 10))

        ttk.Button(btn_frame, text="Calibrate / Edit…", command=self._edit_selected_word).pack(
            side="left", padx=4
        )
        ttk.Button(
            btn_frame, text="Delete Samples", command=self._delete_selected_word_samples
        ).pack(side="left", padx=4)

        ttk.Button(btn_frame, text="Export Profile…", command=self._export_profile).pack(
            side="right", padx=4
        )
        ttk.Button(btn_frame, text="Import Profile…", command=self._import_profile).pack(
            side="right", padx=4
        )
        ttk.Button(btn_frame, text="Close", command=win.destroy).pack(
            side="right", padx=4
        )

        self.word_filter_var.trace_add("write", lambda *args: self._refresh_word_table())
        self.word_status_filter_var.trace_add(
            "write", lambda *args: self._refresh_word_table()
        )

        win.protocol("WM_DELETE_WINDOW", win.destroy)
        self._center_window(win, 800, 500)
        self._refresh_word_table()

    def _update_target_samples_from_ui(self):
        val = self.target_samples_var.get()
        if isinstance(val, int) and val > 0:
            self.voice_profile.set_target_samples(val)
            self.voice_profile.save()
            self._refresh_word_table()

    def _refresh_word_table(self):
        if self.word_tree is None:
            return
        tree = self.word_tree
        for item in tree.get_children():
            tree.delete(item)

        text_filter = (self.word_filter_var.get() or "").strip().lower()
        status_filter = (self.word_status_filter_var.get() or "All").strip()

        for word in sorted(self.voice_profile.words.keys()):
            meta = self.voice_profile.words[word]
            status = self.voice_profile.get_status(word)
            aliases = meta["aliases"]
            n = len(aliases)
            last = meta["last_calibrated"] or ""

            if text_filter and text_filter not in word and text_filter not in ", ".join(
                aliases
            ):
                continue
            if status_filter != "All" and status != status_filter:
                continue

            samples_text = f"{n}/{self.voice_profile.target_samples}"
            tree.insert(
                "",
                "end",
                iid=word,
                values=(word, status, samples_text, last),
            )

    def _get_selected_word(self):
        if self.word_tree is None:
            return None
        sel = self.word_tree.selection()
        if not sel:
            messagebox.showinfo("No selection", "Select a word in the table first.")
            return None
        return sel[0]

    def _edit_selected_word(self):
        word = self._get_selected_word()
        if not word:
            return
        self._open_word_edit_dialog(word)

    def _delete_selected_word_samples(self):
        word = self._get_selected_word()
        if not word:
            return
        if not self.voice_profile.words[word]["aliases"]:
            return
        if not messagebox.askyesno(
            "Delete samples",
            f"Delete all aliases/samples for '{word}'?",
            parent=self.word_calib_window,
        ):
            return
        self.voice_profile.delete_aliases(word, "all")
        self.voice_profile.save()
        self._refresh_word_table()

    def _calibrate_all_words(self):
        pending = [
            w
            for w in sorted(self.voice_profile.words.keys())
            if self.voice_profile.get_status(w) in ("Uncalibrated", "Partial")
        ]
        if not pending:
            messagebox.showinfo(
                "Calibrate All",
                "All words already have at least the target number of samples.",
                parent=self.word_calib_window,
            )
            return

        def step(i=0):
            if i >= len(pending):
                self._refresh_word_table()
                return
            word = pending[i]
            self._open_word_edit_dialog(word, on_close=lambda: step(i + 1))

        step(0)

    def _export_profile(self):
        parent = self.word_calib_window or self
        path = filedialog.asksaveasfilename(
            parent=parent,
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="codex_voice_profile.json",
        )
        if not path:
            return
        try:
            self.voice_profile.export_to_file(path)
        except Exception as e:
            messagebox.showerror(
                "Export error",
                f"Could not export profile:\n{e}",
                parent=parent,
            )
            return
        messagebox.showinfo(
            "Export complete",
            f"Calibration profile saved to:\n{path}",
            parent=parent,
        )

    def _import_profile(self):
        parent = self.word_calib_window or self
        path = filedialog.askopenfilename(
            parent=parent,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.voice_profile.import_from_file(path, replace=True)
            self.voice_profile.save()
        except Exception as e:
            messagebox.showerror(
                "Import error",
                f"Could not import profile:\n{e}",
                parent=parent,
            )
            return
        self._refresh_word_table()
        messagebox.showinfo("Import complete", "Calibration profile loaded.", parent=parent)

    def _on_word_edit_voice_toggle(self, word: str):
        if self.word_edit_voice_enabled is None:
            return
        enabled = bool(self.word_edit_voice_enabled.get())
        if enabled:
            if self.listening:
                self.voice_mode = f"word_calib:{word}"
                self._append_console(
                    f"[VOICE] Word calibration: capturing mic phrases for '{word}'.\n"
                )
            else:
                self._append_console(
                    "[VOICE] Microphone is not listening; turn Listening ON to capture.\n"
                )
                self.word_edit_voice_enabled.set(False)
        else:
            if isinstance(self.voice_mode, str) and self.voice_mode.startswith("word_calib"):
                self.voice_mode = "idle"
                self._append_console(
                    "[VOICE] Word calibration: stopped capturing mic phrases.\n"
                )

    def _open_word_edit_dialog(self, word: str, on_close=None):
        self.voice_profile.ensure_word(word)

        if self.word_edit_window is not None and self.word_edit_window.winfo_exists():
            try:
                self.word_edit_window.destroy()
            except Exception:
                pass

        win = tk.Toplevel(self)
        win.title(f"Calibrate '{word}'")
        self.word_edit_window = win
        self.word_edit_word = word
        self._center_window(win, 500, 360)

        ttk.Label(
            win,
            text=(
                f"Calibrating word: '{word}'\n\n"
                "Each line below is one sample / alias — for example, how Vosk actually\n"
                "prints what it heard when you say this word (e.g., 'guy did' for 'guided').\n"
                "Add or edit lines, then click Save. You can also capture phrases by voice."
            ),
            justify="left",
            wraplength=460,
        ).pack(padx=10, pady=(10, 6), anchor="w")

        text = tk.Text(win, height=8, width=50)
        aliases = self.voice_profile.get_aliases(word)
        if aliases:
            text.insert("1.0", "\n".join(aliases))
        text.pack(padx=10, pady=(0, 8), fill="both", expand=True)
        self.word_edit_text = text

        info_var = tk.StringVar()
        info_label = ttk.Label(win, textvariable=info_var, justify="left")
        info_label.pack(padx=10, pady=(0, 4), anchor="w")
        self.word_edit_info_var = info_var

        self.word_edit_voice_enabled = tk.BooleanVar(value=self.listening)
        cb = ttk.Checkbutton(
            win,
            text="Capture phrases from microphone while this window is open",
            variable=self.word_edit_voice_enabled,
            command=lambda: self._on_word_edit_voice_toggle(word),
        )
        cb.pack(padx=10, pady=(0, 4), anchor="w")

        def update_info():
            lines = [
                l.strip()
                for l in text.get("1.0", tk.END).splitlines()
                if l.strip()
            ]
            n = len(lines)
            target = self.voice_profile.target_samples
            status = (
                "Uncalibrated"
                if n == 0
                else "Partial"
                if n < target
                else "Calibrated"
            )
            info_var.set(f"Samples: {n}/{target}   Status if saved: {status}")

        update_info()

        def on_text_change(event):
            text.edit_modified(False)
            win.after(10, update_info)

        text.bind("<<Modified>>", on_text_change)

        # If listening is already on, auto-enable capture by default
        if self.listening and self.word_edit_voice_enabled.get():
            self.voice_mode = f"word_calib:{word}"
            self._append_console(
                f"[VOICE] Word calibration: capturing mic phrases for '{word}'.\n"
            )

        btn_frame = ttk.Frame(win)
        btn_frame.pack(pady=(4, 10))

        def close_common():
            if isinstance(self.voice_mode, str) and self.voice_mode.startswith("word_calib"):
                self.voice_mode = "idle"
            self.word_edit_window = None
            self.word_edit_text = None
            self.word_edit_word = None
            self.word_edit_info_var = None
            self.word_edit_voice_enabled = None
            if on_close:
                on_close()

        def save_and_close():
            lines = [
                l.strip()
                for l in text.get("1.0", tk.END).splitlines()
                if l.strip()
            ]
            self.voice_profile.set_aliases(word, lines)
            self.voice_profile.save()
            if self.word_calib_window is not None and self.word_calib_window.winfo_exists():
                self._refresh_word_table()
            win.destroy()
            close_common()

        def cancel_and_close():
            win.destroy()
            close_common()

        ttk.Button(btn_frame, text="Save", command=save_and_close).pack(
            side="left", padx=5
        )
        ttk.Button(btn_frame, text="Cancel", command=cancel_and_close).pack(
            side="left", padx=5
        )

        win.grab_set()
        win.focus_set()
        win.protocol("WM_DELETE_WINDOW", cancel_and_close)

    def _handle_word_calib_heard(self, word: str, raw: str, lower: str):
        if self.word_edit_window is None or not self.word_edit_window.winfo_exists():
            self.voice_mode = "idle"
            return
        if self.word_edit_word != word:
            return
        if self.word_edit_text is None:
            return

        txt = self.word_edit_text
        existing = txt.get("1.0", tk.END).strip()
        if existing:
            txt.insert(tk.END, "\n" + raw)
        else:
            txt.insert("1.0", raw)
        txt.see(tk.END)

        if self.word_edit_info_var is not None:
            lines = [
                l.strip()
                for l in txt.get("1.0", tk.END).splitlines()
                if l.strip()
            ]
            n = len(lines)
            target = self.voice_profile.target_samples
            status = (
                "Uncalibrated"
                if n == 0
                else "Partial"
                if n < target
                else "Calibrated"
            )
            self.word_edit_info_var.set(f"Samples: {n}/{target}   Status if saved: {status}")

        self._append_console(f"[CALIB-WORD {word}] {raw}\n")

    # -------------------------------------------------------------------------
    # Core Codex run logic
    # -------------------------------------------------------------------------
    def on_run(self):
        prompt = self.prompt_text.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showwarning("No prompt", "Please enter a prompt first.")
            return

        model = self.model_var.get()
        reasoning = self.reasoning_var.get()

        # log prompt if enabled
        self._pr_log(
            f"[{datetime.now().isoformat()}] PROMPT model={model} reasoning={reasoning}\n{prompt}\n"
        )

        self.run_button.config(state=tk.DISABLED)
        self.status_var.set("Running Codex…")

        self.output_text.config(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, f"Model: {model}\nReasoning: {reasoning}\n\n")

        self._append_console("\n" + "=" * 80 + "\n")
        self._append_console("New run:\n")
        self._append_console(f"Model: {model}, Reasoning: {reasoning}\n")
        self._append_console(f"Prompt:\n{prompt}\n\n")

        thread = threading.Thread(
            target=self._run_codex,
            args=(prompt, model, reasoning),
            daemon=True,
        )
        thread.start()

    def _run_codex(self, prompt: str, model: str, reasoning: str):
        cmd = [
            "codex",
            "exec",
            "--skip-git-repo-check",
            "--model",
            model,
            "--config",
            f'model_reasoning_effort="{reasoning}"',
            prompt,
        ]

        cmd_str = " ".join(
            f'"{c}"' if " " in c and not c.startswith("--") else c for c in cmd
        )
        self.after(0, self._append_console, f"Command:\n{cmd_str}\n\n")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                output = result.stdout
            else:
                output = (
                    f"Error (exit code {result.returncode}):\n\n"
                    f"{result.stderr or '(no stderr)'}"
                )
        except FileNotFoundError:
            output = (
                "Error: 'codex' CLI not found.\n\n"
                "Make sure you have installed it and it's on your PATH:\n"
                "  npm install -g @openai/codex\n"
                "Then run:\n"
                "  codex login\n"
                "to authenticate with your ChatGPT account."
            )
        except Exception as e:
            output = f"Unexpected error:\n{e}"

        self.after(0, self._update_output_and_console, output)

    def _update_output_and_console(self, text: str):
        self.output_text.config(state="normal")
        self.output_text.insert(tk.END, text)

        self._append_console("Output:\n")
        self._append_console(text + "\n")

        # log response if enabled
        self._pr_log(
            f"[{datetime.now().isoformat()}] RESPONSE model={self.model_var.get()} "
            f"reasoning={self.reasoning_var.get()}\n{text}\n"
        )

        try:
            self.bell()
        except Exception:
            pass
        self._append_console("[VOICE] Response ready (ding).\n")

        self.run_button.config(state=tk.NORMAL)
        self.status_var.set("Done.")

    def _append_console(self, text: str):
        if not hasattr(self, "console_text"):
            return
        self.console_text.config(state="normal")
        self.console_text.insert(tk.END, text)
        self.console_text.see(tk.END)
        self.console_text.config(state="disabled")

        if self.debug_logging_var.get():
            self._debug_log(text)

    # -------------------------------------------------------------------------
    # Voice / listening control (Vosk)
    # -------------------------------------------------------------------------
    def toggle_listening(self):
        if not VOSK_AVAILABLE:
            messagebox.showerror(
                "Voice unavailable",
                "Vosk and/or sounddevice not installed.\n\n"
                "Install with:\n"
                "  pip install vosk sounddevice\n"
                "and download a Vosk model, then update VOSK_MODEL_PATH.",
            )
            return

        if not self.listening:
            self.start_listening()
        else:
            self.stop_listening()

    def start_listening(self):
        if self.listening:
            return

        if self.vosk_model is None:
            try:
                self._append_console(f"[VOICE] Loading Vosk model from: {VOSK_MODEL_PATH}\n")
                self.vosk_model = Model(VOSK_MODEL_PATH)
            except Exception as e:
                messagebox.showerror(
                    "Vosk model error",
                    f"Could not load Vosk model at '{VOSK_MODEL_PATH}'.\n\n{e}",
                )
                return

        self.vosk_recognizer = KaldiRecognizer(self.vosk_model, self.sample_rate)

        def audio_callback(indata, frames, time_info, status):
            if status:
                print(status, file=sys.stderr)
            self.audio_queue.put(bytes(indata))

        try:
            self.audio_stream = sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=8000,
                dtype="int16",
                channels=1,
                callback=audio_callback,
            )
            self.audio_stream.start()
        except Exception as e:
            messagebox.showerror("Audio error", f"Could not start audio input:\n{e}")
            return

        self.listening = True
        self.listen_button.config(text="🎤 Listening: ON")
        self.status_var.set("Listening… Say your assistant name to issue commands.")
        self._append_console("[VOICE] Listening started.\n")

        self.voice_thread = threading.Thread(target=self._voice_loop, daemon=True)
        self.voice_thread.start()

    def stop_listening(self):
        if not self.listening:
            return
        self.listening = False
        if self.audio_stream is not None:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
            except Exception:
                pass
            self.audio_stream = None
        self.listen_button.config(text="🎤 Listening: OFF")
        self.status_var.set("Ready.")
        self._append_console("[VOICE] Listening stopped.\n")

    def _voice_loop(self):
        while self.listening:
            try:
                data = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if not self.listening:
                break

            if self.vosk_recognizer.AcceptWaveform(data):
                result_json = self.vosk_recognizer.Result()
                try:
                    result = json.loads(result_json)
                    text = result.get("text", "").strip()
                except Exception:
                    text = ""
                if text:
                    self.after(0, self._handle_voice_text, text)

    # -------------------------------------------------------------------------
    # Voice command handling
    # -------------------------------------------------------------------------
    def _handle_voice_text(self, text: str):
        raw_orig = text.strip()
        if not raw_orig:
            return

        raw_tokens = raw_orig.split()
        lower_tokens = [t.lower() for t in raw_tokens]
        FILLERS = {"huh", "uh", "um", "erm", "mm", "mmm", "uhh", "uhm"}

        start = 0
        end = len(raw_tokens)
        while start < end and lower_tokens[start].strip(" ,.!?'\"") in FILLERS:
            start += 1
        while end > start and lower_tokens[end - 1].strip(" ,.!?'\"") in FILLERS:
            end -= 1

        if start >= end:
            return

        raw_tokens = raw_tokens[start:end]
        lower_tokens = lower_tokens[start:end]

        raw = " ".join(raw_tokens)
        lower = " ".join(t.strip(" ,.!?'\"") for t in lower_tokens)

        self._append_console(f"[VOICE] Heard: {raw}\n")

        # Confirm-clear mode
        if self.voice_mode == "confirm_clear":
            tokens = lower.split()

            yes_targets = ["confirm", "yes", "yeah", "yep"]
            no_targets = ["cancel", "no", "nope"]

            if "yes" in self.voice_profile.words:
                yes_targets.extend(self.voice_profile.get_aliases("yes"))
            if "no" in self.voice_profile.words:
                no_targets.extend(self.voice_profile.get_aliases("no"))

            if any(self._fuzzy(t, yes_targets, threshold=0.75) for t in tokens):
                self._append_console("[VOICE] Clear confirmed by voice.\n")
                self._finish_clear(True)
                return
            if any(self._fuzzy(t, no_targets, threshold=0.75) for t in tokens):
                self._append_console("[VOICE] Clear cancelled by voice.\n")
                self._finish_clear(False)
                return
            return

        # Global calibration mode
        if self.voice_mode == "calibrate":
            self.calibration_captured.append(lower)
            self._append_console(f"[CALIB] {raw}\n")
            self._append_calibration_heard(raw)
            return

        # Per-word calibration mode
        if isinstance(self.voice_mode, str) and self.voice_mode.startswith("word_calib:"):
            word = self.voice_mode.split(":", 1)[1]
            self._handle_word_calib_heard(word, raw, lower)
            return

        # Prompt capture mode
        if self.voice_mode == "prompt":
            tokens = lower.split()

            if len(tokens) >= 2:
                first = tokens[0].strip(",.!?")
                second = tokens[1].strip(",.!?")
                if self._fuzzy(first, self._wake_targets_lower(), threshold=0.72) and self._fuzzy(
                    second, ["enter", "end"]
                ):
                    self.voice_mode = "idle"
                    self._append_console("[VOICE] Ending prompt capture via wake command.\n")
                    self._guided_say("Prompt capture ended.")
                    return

            if tokens:
                first = tokens[0].strip(",.!?")
                if self._fuzzy(first, self._wake_targets_lower(), threshold=0.72):
                    handled = self._handle_wake_command(lower)
                    if handled:
                        self._append_console("[VOICE] Prompt mode: handled assistant command.\n")
                    else:
                        self._append_console(
                            "[VOICE] Prompt mode: ignored unknown assistant command.\n"
                        )
                        self._guided_say("I didn't recognize that command.")
                    return

            self._append_to_prompt(raw + " ")
            return

        # Idle mode
        if self._handle_wake_command(lower):
            return

    # -------------------------------------------------------------------------
    # Wake-name command dispatcher
    # -------------------------------------------------------------------------
    def _guided_aliases(self):
        return self.voice_profile.get_aliases("guided")

    def _unguided_aliases(self):
        return self.voice_profile.get_aliases("unguided")

    def _handle_wake_command(self, lower: str) -> bool:
        tokens = lower.split()
        if not tokens:
            return False

        wake_targets = self._wake_targets_lower()
        base_name = self._get_wake_name_lower()

        first = tokens[0].strip(",.!?")
        if not self._fuzzy(first, wake_targets, threshold=0.72):
            return False

        cmd = " ".join(tokens[1:]).strip()
        if not cmd:
            return False

        cmd_tokens = cmd.split()
        head = cmd_tokens[0] if cmd_tokens else ""
        rest = " ".join(cmd_tokens[1:]).strip()

        # Stop listening
        if self._fuzzy(head, ["stop"]) and "listen" in cmd:
            self._append_console("[VOICE] Command: stop listening.\n")
            self._guided_say("Stopping listening.")
            self.stop_listening()
            return True

        # Mode
        if self._fuzzy(head, ["mode", "node", "note", "mowed", "mold", "modee"], threshold=0.70):
            rem = rest.replace("-", " ")
            rem_low = rem.lower()
            tokens_rem = rem_low.split()

            guided_targets = ["guided"] + self._guided_aliases()
            unguided_targets = ["unguided"] + self._unguided_aliases()

            guided_match = any(
                self._fuzzy(t, guided_targets, threshold=0.78) for t in tokens_rem
            )
            unguided_match_token = any(
                self._fuzzy(t, unguided_targets, threshold=0.78) for t in tokens_rem
            )
            explicit_un_guided = ("unguided" in rem_low) or ("un guided" in rem_low)

            self._append_console(f"[VOICE] Command: mode ({rem}).\n")

            if explicit_un_guided or (unguided_match_token and not guided_match):
                self.mode_var.set("Unguided")
                self._append_console("[VOICE] Mode set to Unguided.\n")
            elif guided_match:
                self.mode_var.set("Guided")
                self._append_console("[VOICE] Mode set to Guided.\n")
                self._guided_say("Guided mode enabled. I will talk you through commands.")
            else:
                self._append_console(
                    "[VOICE] Could not understand mode setting. Say 'mode guided' or 'mode unguided'.\n"
                )
                self._guided_say(
                    "I did not catch which mode. Say mode guided or mode unguided."
                )
            return True

        # Prompt
        if self._fuzzy(head, ["prompt"]):
            self._append_console("[VOICE] Command: prompt (start capture).\n")
            self.voice_mode = "prompt"
            if rest:
                self._append_to_prompt(rest + " ")
            self._guided_say("Prompt capture started. Begin speaking your prompt.")
            return True

        # Model
        if self._fuzzy(head, ["model"]):
            self._append_console(f"[VOICE] Command: model ({rest}).\n")
            self._handle_voice_model(rest)
            return True

        # Reasoning
        if self._fuzzy(head, ["reasoning", "reason"]):
            self._append_console(f"[VOICE] Command: reasoning ({rest}).\n")
            self._handle_voice_reasoning(rest)
            return True

        # Respond
        if self._fuzzy(head, ["respond", "reply", "answer"]):
            self._append_console("[VOICE] Command: respond (speak response).\n")

            if self.tts_paused:
                self._append_console("[VOICE] Respond while paused -> continuing speech.\n")
                self._continue_tts()
                return True

            if self.tts_thread is not None and self.tts_thread.is_alive():
                self._append_console(
                    "[VOICE] Response already being spoken; say stop, then respond again.\n"
                )
                return True

            self._start_tts_read_response()
            return True

        # Pause speech
        if self._fuzzy(head, ["pause"]):
            self._append_console("[VOICE] Command: pause speech.\n")
            self._pause_tts()
            return True

        # Continue speech
        if self._fuzzy(head, ["continue", "resume"]):
            self._append_console("[VOICE] Command: continue speech.\n")
            self._continue_tts()
            return True

        # Stop speech
        if self._fuzzy(head, ["stop"]):
            self._append_console("[VOICE] Command: stop speech.\n")
            self._stop_tts_command()
            return True

        # Help
        if self._fuzzy(head, ["help"]):
            self._append_console("[VOICE] Command: help.\n")
            self._speak_help()
            return True

        # Run
        if self._fuzzy(head, ["run", "go", "execute"]):
            self._append_console("[VOICE] Command: run.\n")
            self._guided_say("Running your prompt.")
            self.on_run()
            return True

        # Clear
        if self._fuzzy(head, ["clear"]):
            self._append_console("[VOICE] Command: clear (needs confirmation).\n")
            self._ask_clear_confirm()
            return True

        # Model previous/next
        if self._fuzzy(head, ["previous", "prev", "back"]):
            self._append_console("[VOICE] Command: model previous.\n")
            self._cycle_model(-1)
            return True
        if self._fuzzy(head, ["next", "forward"]):
            self._append_console("[VOICE] Command: model next.\n")
            self._cycle_model(1)
            return True

        self._append_console(f"[VOICE] Unknown '{base_name}' command: {cmd}\n")
        self._guided_say("I didn't recognize that command.")
        return False

    # --- Model & reasoning helpers ------------------------------------------
    def _cycle_model(self, direction: int):
        try:
            current = self.model_var.get()
            idx = MODEL_OPTIONS.index(current)
        except ValueError:
            idx = 0
        new_idx = (idx + direction) % len(MODEL_OPTIONS)
        self.model_var.set(MODEL_OPTIONS[new_idx])
        self._append_console(f"[VOICE] Model set to: {self.model_var.get()}\n")
        self._guided_say(f"Model set to {self.model_var.get()}.")

    def _handle_voice_model(self, remainder: str):
        rem = remainder.strip()
        if not rem:
            return

        if "previous" in rem or "back" in rem:
            self._cycle_model(-1)
            return
        if "next" in rem or "forward" in rem:
            self._cycle_model(1)
            return

        spoken = rem.replace("-", " ").replace("_", " ")
        for model in MODEL_OPTIONS:
            m_norm = model.replace("-", " ").replace("_", " ").lower()
            if m_norm in spoken:
                self.model_var.set(model)
                self._append_console(f"[VOICE] Model set to: {model}\n")
                self._guided_say(f"Model set to {model}.")
                return

        self._append_console("[VOICE] Could not match model name.\n")
        self._guided_say("I could not match that model name.")

    def _cycle_reasoning(self, direction: int):
        try:
            current = self.reasoning_var.get()
            idx = REASONING_OPTIONS.index(current)
        except ValueError:
            idx = 1
        new_idx = (idx + direction) % len(REASONING_OPTIONS)
        self.reasoning_var.set(REASONING_OPTIONS[new_idx])
        self._append_console(f"[VOICE] Reasoning set to: {self.reasoning_var.get()}\n")
        self._guided_say(f"Reasoning set to {self.reasoning_var.get()}.")

    def _handle_voice_reasoning(self, remainder: str):
        rem = remainder.strip()
        if not rem:
            return

        rem = rem.replace("degrees", "decrease")

        if "increase" in rem or "up" in rem or "higher" in rem:
            self._cycle_reasoning(1)
            return
        if "decrease" in rem or "down" in rem or "lower" in rem:
            self._cycle_reasoning(-1)
            return

        for level in REASONING_OPTIONS:
            if level in rem:
                self.reasoning_var.set(level)
                self._append_console(f"[VOICE] Reasoning set to: {level}\n")
                self._guided_say(f"Reasoning set to {level}.")
                return

        self._append_console("[VOICE] Could not match reasoning level.\n")
        self._guided_say("I could not match that reasoning level.")

    # --- Prompt helpers ------------------------------------------------------
    def _append_to_prompt(self, text: str):
        self.prompt_text.insert(tk.END, text)
        self.prompt_text.see(tk.END)

    # --- Clear with confirm --------------------------------------------------
    def _ask_clear_confirm(self):
        if (
            self.clear_confirm_window is not None
            and self.clear_confirm_window.winfo_exists()
        ):
            return

        self.voice_mode = "confirm_clear"

        win = tk.Toplevel(self)
        win.title("Confirm clear")
        self.clear_confirm_window = win

        label = ttk.Label(
            win,
            text=(
                "Clear prompt, response, and console?\n\n"
                "Say 'confirm' or 'cancel', or click a button."
            ),
            justify="center",
        )
        label.pack(padx=20, pady=15)

        btn_frame = ttk.Frame(win)
        btn_frame.pack(pady=(0, 15))

        ttk.Button(btn_frame, text="Confirm", command=lambda: self._finish_clear(True)).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(btn_frame, text="Cancel", command=lambda: self._finish_clear(False)).pack(
            side=tk.LEFT, padx=5
        )

        self._center_window(win, width=360, height=160)

        self._guided_say(
            "Do you want me to clear the prompt, response, and console? Say confirm or cancel."
        )

    def _finish_clear(self, confirmed: bool):
        if confirmed:
            self._do_clear()
            self._guided_say("Cleared all panes.")
        else:
            self._append_console("[VOICE] Clear cancelled.\n")
            self._guided_say("Clear cancelled.")

        self.voice_mode = "idle"

        if self.clear_confirm_window is not None:
            try:
                self.clear_confirm_window.destroy()
            except Exception:
                pass
            self.clear_confirm_window = None

    def _do_clear(self):
        self.prompt_text.delete("1.0", tk.END)
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.console_text.config(state="normal")
        self.console_text.delete("1.0", tk.END)
        self.console_text.config(state="disabled")
        self._append_console("[VOICE] Cleared all panes.\n")

    # --- Misc helpers --------------------------------------------------------
    def _center_window(self, win: tk.Toplevel, width: int, height: int):
        win.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() // 2) - (width // 2)
        y = self.winfo_y() + (self.winfo_height() // 2) - (height // 2)
        win.geometry(f"{width}x{height}+{x}+{y}")

    def on_close(self):
        self.stop_listening()
        self.tts_should_stop = True
        if self.debug_log_file:
            try:
                self.debug_log_file.close()
            except Exception:
                pass
            self.debug_log_file = None
        if self.pr_log_file:
            try:
                self.pr_log_file.close()
            except Exception:
                pass
            self.pr_log_file = None
        self.destroy()


def main():
    app = CodexVoiceGUI()
    app.mainloop()


if __name__ == "__main__":
    main()


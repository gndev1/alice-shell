#!/usr/bin/env python3
"""
alice-shell.py - Voice-enabled CLI shell for Codex / GPT-5

Features:
- Voice + typed control of Codex / GPT models
- "Shell command" suggestions from buffer + prompt
- "Voice command" suggestions for internal v- / Alice commands
- Smart execution of suggested commands:
  - "v-..." => internal command in this app
  - "<AssistantName> ..." => voice command in this app
  - anything else => OS shell command (with confirmation)
- Experimental self-directed test mode:
  - "Alice test voice" / "Alice test shell" / "Alice test both"
"""

import os
import sys
import json
import time
import queue
import shlex
import threading
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# Optional imports for STT / TTS / color
try:
    import sounddevice as sd
except Exception:
    sd = None

try:
    from vosk import Model, KaldiRecognizer
except Exception:
    Model = None
    KaldiRecognizer = None

try:
    from TTS.api import TTS
except Exception:
    TTS = None

try:
    import soundfile as sf
except Exception:
    sf = None

try:
    from colorama import init as colorama_init, Fore, Style
except Exception:
    colorama_init = None
    Fore = None
    Style = None

# Defaults
DEFAULT_MODEL = "gpt-5"
DEFAULT_REASONING = "medium"

VOSK_MODEL_DIR = "vosk-model-small-en-us-0.15"
COQUI_MODEL_NAME = "tts_models/en/vctk/vits"

ASSISTANT_DEFAULT_NAME = "Alice"

RECORDINGS_DIR = "recordings"
LOG_DIR = "logs"

PROFILE_FILENAME = "codex_voice_profile.json"
SETTINGS_FILENAME = "codex_shell_settings.json"

SHELL_DEBUG_LOG = "codex_shell_debug.log"
SHELL_SESSION_LOG = "codex_shell_session.log"

TTS_SPEED_MIN = 0.5
TTS_SPEED_MAX = 2.0
TTS_SPEED_STEP = 0.25


def safe_print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


@dataclass
class VoiceProfile:
    """Simple alias / calibration profile persisted as JSON."""
    profile_path: Path
    data: Dict[str, List[str]] = field(default_factory=dict)

    def load(self):
        if self.profile_path.exists():
            try:
                with self.profile_path.open("r", encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, dict):
                    self.data = {k: [w.lower() for w in v if isinstance(w, str)] for k, v in obj.items()}
                    safe_print(f"[CALIB] Loaded voice profile from {self.profile_path}")
            except Exception as e:
                safe_print(f"[CALIB] Failed to load profile {self.profile_path}: {e}")

    def save(self):
        try:
            self.profile_path.parent.mkdir(parents=True, exist_ok=True)
            with self.profile_path.open("w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
            safe_print(f"[CALIB] Saved voice profile to {self.profile_path}")
        except Exception as e:
            safe_print(f"[CALIB] Failed to save profile {self.profile_path}: {e}")

    def get_aliases(self, key: str, default: Optional[List[str]] = None) -> List[str]:
        vals = self.data.get(key.lower())
        if not vals:
            return list(default or [])
        seen = set()
        out: List[str] = []
        for w in vals:
            wl = w.lower().strip()
            if wl and wl not in seen:
                seen.add(wl)
                out.append(wl)
        return out

    def add_aliases(self, key: str, words: List[str]):
        key = key.lower()
        base = self.data.get(key, [])
        base = [b.lower() for b in base]
        for w in words:
            wl = w.lower().strip()
            if wl and wl not in base:
                base.append(wl)
        self.data[key] = base


class ShellLogger:
    def __init__(self, enable_debug: bool, enable_session_log: bool, base_dir: Path):
        self.enable_debug = enable_debug
        self.enable_session_log = enable_session_log
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.debug_path = self.base_dir / SHELL_DEBUG_LOG
        self.session_path = self.base_dir / SHELL_SESSION_LOG
        self._debug_lock = threading.Lock()
        self._sess_lock = threading.Lock()

    def debug(self, msg: str):
        if not self.enable_debug:
            return
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}\n"
        with self._debug_lock:
            try:
                with self.debug_path.open("a", encoding="utf-8") as f:
                    f.write(line)
            except Exception:
                pass

    def session(self, msg: str):
        if not self.enable_session_log:
            return
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}\n"
        with self._sess_lock:
            try:
                with self.session_path.open("a", encoding="utf-8") as f:
                    f.write(line)
            except Exception:
                pass


class CodexVoiceShell:
    def __init__(self, args):
        self.script_dir = Path(__file__).resolve().parent

        # Logger (flags will be updated after settings load)
        self.logger = ShellLogger(False, False, self.script_dir / LOG_DIR)

        # --- Default settings (as requested) ---
        self.assistant_name = ASSISTANT_DEFAULT_NAME
        self.guided_mode = True
        self.current_model = DEFAULT_MODEL
        self.current_reasoning = DEFAULT_REASONING
        self.debug_enabled = True
        self.save_recordings = True
        self.save_prompts = True
        self.use_color = True
        self.tts_speed = 1.20
        self.buffer_mode = "session"  # session / anchor / last
        self.buffer_anchor_index = 0
        self.buffer_last_action_index = 0
        self.voice_index = 1  # 0-based index; 1 => second voice

        # CLI overrides that we apply after loading settings
        cli_overrides = {}

        for a in args:
            al = a.lower()
            if al == "-logdebug":
                cli_overrides["debug_enabled"] = True
            if al == "-nodebug":
                cli_overrides["debug_enabled"] = False
            if al == "-recordresponsespeech":
                cli_overrides["save_recordings"] = True
            if al == "-norecordresponsespeech":
                cli_overrides["save_recordings"] = False
            if al == "-saveprompts":
                cli_overrides["save_prompts"] = True
            if al == "-nosaveprompts":
                cli_overrides["save_prompts"] = False
            if al in ("-fancy", "-ui", "-color"):
                cli_overrides["use_color"] = True
            if al == "-nofancy":
                cli_overrides["use_color"] = False
            if al == "-guided":
                cli_overrides["guided_mode"] = True
            if al == "-unguided":
                cli_overrides["guided_mode"] = False
            if al.startswith("-voice:"):
                try:
                    idx = int(a.split(":", 1)[1].strip())
                    if idx > 0:
                        cli_overrides["voice_index"] = idx - 1
                except Exception:
                    pass
            if al.startswith("-speed:"):
                try:
                    val = float(a.split(":", 1)[1].strip())
                    cli_overrides["tts_speed"] = max(TTS_SPEED_MIN, min(TTS_SPEED_MAX, val))
                except Exception:
                    pass

        # Apply saved settings (if present) then CLI overrides
        self.settings_path = self.script_dir / SETTINGS_FILENAME
        self._load_settings()
        for k, v in cli_overrides.items():
            setattr(self, k, v)

        # Now configure logger flags
        self.logger.enable_debug = self.debug_enabled
        self.logger.enable_session_log = self.save_prompts

        # Color
        if self.use_color and colorama_init and Fore and Style:
            colorama_init()
        else:
            self.use_color = False

        # Voice profile
        profile_path = self.script_dir / PROFILE_FILENAME
        self.profile = VoiceProfile(profile_path=profile_path)
        self.profile.load()

        # State
        self.terminal_buffer: List[str] = []
        self.prompt_text: str = ""
        self.pending_shell_command: Optional[str] = None
        self.pending_shell_explanation: Optional[str] = None
        self.last_codex_prompt: Optional[str] = None
        self.last_codex_response: Optional[str] = None

        self.prompt_capture = False
        self.confirm_mode: Optional[str] = None  # "clear" / "exec" / "exit" / "save_settings" / ...

        self._exit_requested = False

        # Audio & STT/TTS
        self._print_lock = threading.Lock()
        self._listening = False
        self._vosk_model: Optional[Model] = None
        self._recognizer: Optional[KaldiRecognizer] = None
        self._audio_queue: "queue.Queue[bytes]" = queue.Queue()
        self._audio_thread: Optional[threading.Thread] = None
        self._stt_thread: Optional[threading.Thread] = None

        self._tts: Optional[TTS] = None
        self._tts_speakers: List[str] = []
        self._tts_lock = threading.Lock()

        # Recordings dir
        self.recordings_dir = self.script_dir / RECORDINGS_DIR
        self.recordings_dir.mkdir(parents=True, exist_ok=True)

        # Init audio
        self._init_tts()
        self._init_stt()

    # --- Settings persistence --------------------------------------------

    def _load_settings(self):
        if not self.settings_path.exists():
            return
        try:
            with self.settings_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return
        if not isinstance(data, dict):
            return
        self.assistant_name = data.get("assistant_name", self.assistant_name)
        self.guided_mode = bool(data.get("guided_mode", self.guided_mode))
        self.current_model = data.get("model", self.current_model)
        self.current_reasoning = data.get("reasoning", self.current_reasoning)
        self.debug_enabled = bool(data.get("debug_enabled", self.debug_enabled))
        self.save_recordings = bool(data.get("save_recordings", self.save_recordings))
        self.save_prompts = bool(data.get("save_prompts", self.save_prompts))
        self.use_color = bool(data.get("use_color", self.use_color))
        try:
            spd = float(data.get("tts_speed", self.tts_speed))
            self.tts_speed = max(TTS_SPEED_MIN, min(TTS_SPEED_MAX, spd))
        except Exception:
            pass
        self.buffer_mode = data.get("buffer_mode", self.buffer_mode)
        try:
            vix = int(data.get("voice_index", self.voice_index))
            if vix >= 0:
                self.voice_index = vix
        except Exception:
            pass

    def _save_settings(self):
        data = {
            "assistant_name": self.assistant_name,
            "guided_mode": self.guided_mode,
            "model": self.current_model,
            "reasoning": self.current_reasoning,
            "debug_enabled": self.debug_enabled,
            "save_recordings": self.save_recordings,
            "save_prompts": self.save_prompts,
            "use_color": self.use_color,
            "tts_speed": float(self.tts_speed),
            "buffer_mode": self.buffer_mode,
            "voice_index": int(self.voice_index),
        }
        try:
            with self.settings_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self._print(f"[SHELL] Settings saved to {self.settings_path}", buffer=False)
        except Exception as e:
            self._print(f"[SHELL] Failed to save settings: {e}", buffer=False)

    # --- Basic printing & status -----------------------------------------

    def _colorize(self, text: str) -> str:
        if not self.use_color or not Fore or not Style:
            return text
        stripped = text.lstrip()
        color = None
        if stripped.startswith("[VOICE]"):
            color = Fore.CYAN
        elif stripped.startswith("[SHELL]"):
            color = Fore.YELLOW
        elif stripped.startswith("[CMD]"):
            color = Fore.GREEN
        elif stripped.startswith("[CONFIRM]"):
            color = Fore.MAGENTA
        elif stripped.startswith("[CALIB]"):
            color = Fore.BLUE
        elif stripped.startswith("[RESP]"):
            color = Fore.LIGHTBLUE_EX
        elif stripped.startswith("[HISTORY]") or stripped.startswith("[H]"):
            color = Fore.WHITE
        elif stripped.startswith("[STATUS]"):
            color = Fore.LIGHTBLACK_EX
        elif stripped.startswith("[TEST]"):
            color = Fore.LIGHTMAGENTA_EX
        elif stripped.startswith("===") or stripped.startswith("┌") or stripped.startswith("└") or stripped.startswith("╔"):
            color = Fore.WHITE + Style.BRIGHT
        if color:
            return color + text + Style.RESET_ALL
        return text

    def _print(self, text: str = "", end: str = "\n", buffer: bool = True):
        with self._print_lock:
            out = self._colorize(text)
            safe_print(out, end=end)
            if buffer and text:
                self.terminal_buffer.append(text)
            self.logger.debug(text)

    def _status_text(self) -> str:
        parts = [
            f"guided={'ON' if self.guided_mode else 'OFF'}",
            f"model={self.current_model}",
            f"reason={self.current_reasoning}",
            f"speed={self.tts_speed:.2f}x",
            f"buffer={self.buffer_mode}",
            f"prompt={'ON' if self.prompt_capture else 'OFF'}",
            f"listen={'ON' if self._listening else 'OFF'}",
        ]
        return " | ".join(parts)

    def _print_status(self):
        self._print(f"[STATUS] {self._status_text()}", buffer=False)

    def _print_banner(self):
        title = "Codex Voice Shell (alice-shell)"
        border = "═" * max(len(title) + 4, 32)
        self._print(f"╔{border}╗")
        centered = title.center(len(border), " ")
        self._print(f"║  {centered}  ║")
        self._print(f"╚{border}╝")
        self._print(f"Assistant name: {self.assistant_name}")
        self._print(f"Model: {self.current_model}, Reasoning: {self.current_reasoning}")
        self._print(f"Guided mode: {'ON' if self.guided_mode else 'OFF'}")
        self._print(f"TTS speed: {self.tts_speed:.2f}x")
        self._print("Type normal shell commands to run them.")
        self._print("Prefix with 'v-' to send commands to the voice assistant (e.g., v-help).")
        self._print("Examples:")
        self._print("  v-help              # show voice shell help")
        self._print("  v-settings          # show current settings")
        self._print("  v-prompt            # edit the Codex prompt via keyboard")
        self._print("  v-command           # ask Codex for next shell command based on buffer")
        self._print("  v-voicecmd          # ask Codex for next internal voice/v- command")
        self._print("  v-history           # show the buffer slice that will be sent")
        self._print("  v-guided-on/off     # toggle guided mode")
        self._print("  v-fancy-on/off      # toggle colored output")
        self._print("  v-speed 1.2         # set TTS speed")
        self._print("  v-buffer clear|session|last   # choose buffer slice for shell command")
        self._print("  v-respond           # read last explanation/response")
        self._print("  v-repeat            # read & print current prompt")
        self._print("  v-recordings        # toggle saving speech recordings")
        self._print("  v-logprompts        # toggle logging prompts/responses")
        self._print("  v-debug             # toggle debug logging")
        self._print("  v-save              # save current settings to disk")
        self._print("  v-exec              # execute last proposed command (v- or shell)")
        self._print("  v-exit              # exit this voice shell")
        self._print("  v-listen            # (re)start microphone listening")
        self._print("Voice commands (with assistant name):")
        self._print("  Alice settings         # read current settings")
        self._print("  Alice save recordings  # enable saving speech recordings (with confirmation)")
        self._print("  Alice log              # enable prompt/response logging (with confirmation)")
        self._print("  Alice debug on/off     # toggle debug logging")
        self._print("  Alice rename <name>    # change assistant name")
        self._print("  Alice prompt           # start prompt editing (voice dictation)")
        self._print("  Alice done             # stop prompt editing (or press Enter on empty line)")
        self._print("  Alice run              # submit the prompt")
        self._print("  Alice shell command    # suggest next OS shell command")
        self._print("  Alice voice command    # suggest next internal v- / Alice command")
        self._print("  Alice execute          # confirm running the suggested command")
        self._print("  Alice respond          # speak explanation / last response")
        self._print("  Alice repeat           # speak & print current prompt")
        self._print("  Alice history          # show buffer slice preview")
        self._print("  Alice buffer clear / session / last")
        self._print("  Alice mode guided / unguided")
        self._print("  Alice speed / speed increase / speed 1.2")
        self._print("  Alice save             # save settings (with confirmation)")
        self._print("  Alice exit             # exit this voice shell (with confirmation)")
        self._print("  Alice listen / stop listening")
        self._print("  Alice test voice / test shell / test both")
        self._print("  Alice self directed    # run both self-tests (voice + shell)")
        self._print()
        self._print_status()

    # --- STT / TTS init --------------------------------------------------

    def _init_stt(self):
        if sd is None or Model is None or KaldiRecognizer is None:
            self._print("[VOICE] Vosk / sounddevice not available; voice disabled.")
            return

        model_path = self.script_dir / VOSK_MODEL_DIR
        if not model_path.exists():
            self._print(f"[VOICE] Vosk model folder not found: {model_path}")
            return

        if self._vosk_model is None or self._recognizer is None:
            try:
                self._print(f"[VOICE] Loading Vosk model from: {model_path}")
                self._vosk_model = Model(str(model_path))
                self._recognizer = KaldiRecognizer(self._vosk_model, 16000)
            except Exception as e:
                self._print(f"[VOICE] Failed to load Vosk model: {e}")
                self._vosk_model = None
                self._recognizer = None
                return

        if self._listening:
            self._print("[VOICE] Already listening.")
            return

        self._listening = True
        self._audio_thread = threading.Thread(target=self._audio_capture_loop, daemon=True)
        self._audio_thread.start()
        self._stt_thread = threading.Thread(target=self._stt_loop, daemon=True)
        self._stt_thread.start()
        self._print("[VOICE] Listening started.")
        self._print_status()

    def _init_tts(self):
        if TTS is None:
            self._print("[VOICE] Coqui TTS not installed; text-to-speech disabled.")
            return
        try:
            self._print(f"[VOICE] Loading Coqui TTS model '{COQUI_MODEL_NAME}' (first run may take a while)...")
            self._tts = TTS(model_name=COQUI_MODEL_NAME, progress_bar=False, gpu=False)
            try:
                speakers = getattr(self._tts, "speakers", None)
                if isinstance(speakers, list) and speakers:
                    self._tts_speakers = speakers
                    if self.voice_index >= len(self._tts_speakers):
                        self.voice_index = 0
                    self._print(f"[VOICE] Coqui speakers: {self._tts_speakers}")
                else:
                    self._tts_speakers = []
                    self._print("[VOICE] Coqui model has no explicit speakers.")
            except Exception:
                self._tts_speakers = []
        except Exception as e:
            self._print(f"[VOICE] Could not initialize Coqui TTS: {e}")
            self._tts = None

    # --- Audio & STT loops -----------------------------------------------

    def _audio_callback(self, indata, frames, time_info, status):
        if not self._listening:
            return
        if status:
            self.logger.debug(f"sounddevice status: {status}")
        try:
            self._audio_queue.put(bytes(indata))
        except Exception as e:
            self.logger.debug(f"audio_queue put error: {e}")

    def _audio_capture_loop(self):
        if sd is None:
            return
        try:
            with sd.RawInputStream(
                samplerate=16000,
                blocksize=8000,
                dtype="int16",
                channels=1,
                callback=self._audio_callback,
            ):
                while self._listening:
                    time.sleep(0.1)
        except Exception as e:
            self._print(f"[VOICE] Audio input error: {e}")

    def _stt_loop(self):
        if self._recognizer is None:
            return
        while self._listening:
            try:
                data = self._audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if self._recognizer.AcceptWaveform(data):
                    txt = self._recognizer.Result()
                    self._handle_vosk_result(txt)
            except Exception as e:
                self.logger.debug(f"stt error: {e}")

    def _handle_vosk_result(self, result_json: str):
        try:
            obj = json.loads(result_json)
        except Exception:
            return
        text = obj.get("text", "").strip()
        if not text:
            return
        if text.lower() in {"huh", "uh", "um"}:
            self.logger.debug(f"[VOICE] Dropped filler: {text}")
            return
        self.logger.debug(f"[VOICE] Heard: {text}")
        self._handle_voice_text(text)

    # --- Voice command parsing -------------------------------------------

    def _norm(self, s: str) -> str:
        return "".join(ch for ch in s.lower().strip() if ch.isalnum() or ch.isspace())

    def _tokenize(self, s: str) -> List[str]:
        return [t for t in self._norm(s).split() if t]

    def _fuzzy_in(self, word: str, candidates: List[str], threshold: float = 0.78) -> bool:
        import difflib
        word = word.lower().strip()
        if not word or not candidates:
            return False
        best = 0.0
        for c in candidates:
            r = difflib.SequenceMatcher(None, word, c.lower().strip()).ratio()
            if r > best:
                best = r
        return best >= threshold

    def _get_wake_variants(self) -> List[str]:
        base = self.assistant_name.lower().strip()
        variants = {base}
        wake_aliases = self.profile.get_aliases("wake", [])
        for w in wake_aliases:
            variants.add(w.lower())
        return sorted(variants)

    def _get_yes_variants(self) -> List[str]:
        return self.profile.get_aliases("yes", ["yes", "yeah", "yep", "confirm", "sure", "ok", "okay"])

    def _get_no_variants(self) -> List[str]:
        return self.profile.get_aliases("no", ["no", "nope", "cancel", "stop", "never"])

    def _get_guided_variants(self) -> List[str]:
        return self.profile.get_aliases("guided", ["guided", "talkative", "explain", "helpful"])

    def _get_unguided_variants(self) -> List[str]:
        return self.profile.get_aliases("unguided", ["unguided", "quiet", "minimal"])

    def _handle_voice_text(self, text: str):
        tokens = self._tokenize(text)
        if not tokens:
            return

        # Confirmation mode overrides everything
        if self.confirm_mode:
            self._handle_confirm_voice(tokens)
            return

        wake_variants = self._get_wake_variants()
        first = tokens[0]

        # Prompt capture mode: anything NOT starting with wake-name is dictation
        if self.prompt_capture and not self._fuzzy_in(first, wake_variants):
            if len(tokens) == 1 and tokens[0] in {"done", "enter"}:
                self.prompt_capture = False
                self._print("[VOICE] Prompt editing finished.")
                if self.guided_mode:
                    self._speak("Prompt editing finished.")
                self._print_status()
                return
            fragment = " ".join(tokens)
            if self.prompt_text:
                self.prompt_text += " "
            self.prompt_text += fragment
            self._print(f"[PROMPT+] {fragment}")
            self._print_status()
            return

        if self._fuzzy_in(first, wake_variants):
            rest = tokens[1:]
            if not rest:
                if self.guided_mode:
                    self._speak(f"Yes? I'm {self.assistant_name}.")
                self._print_status()
                return
            self._handle_wake_command(rest)
        else:
            return

    def _apply_confirm_decision(self, accepted: bool):
        mode = self.confirm_mode
        self.confirm_mode = None

        if not mode:
            return

        if not accepted:
            self._print("[CONFIRM] Cancelled.")
            if self.guided_mode:
                self._speak("Cancelled.")
            self._print_status()
            return

        if mode == "exec" and self.pending_shell_command:
            cmd = self.pending_shell_command.strip()
            self._print(f"[EXEC] Running: {cmd}")
            # Internal v- command
            if cmd.startswith("v-"):
                self._handle_typed_voice_command(cmd)
            else:
                tokens = cmd.split()
                if tokens and tokens[0].lower() == self.assistant_name.lower():
                    # Treat as voice command: "Alice something"
                    self._handle_wake_command(self._tokenize(" ".join(tokens[1:])))
                else:
                    # Plain shell command
                    self._run_shell_command(cmd)
            self.pending_shell_command = None
            self.pending_shell_explanation = None
            if self.guided_mode:
                self._speak("Executed the recommended command.")
        elif mode == "clear":
            self.terminal_buffer.clear()
            self._print("[SHELL] Buffer cleared.")
            if self.guided_mode:
                self._speak("Cleared the buffer.")
        elif mode == "exit":
            self._exit_requested = True
            self._listening = False
            self._print("[SHELL] Exit confirmed.")
            if self.guided_mode:
                self._speak("Exiting voice shell.")
        elif mode == "save_settings":
            self._save_settings()
            if self.guided_mode:
                self._speak("Settings saved.")
        elif mode == "save_recordings":
            self.save_recordings = True
            self._print("[VOICE] Saving recordings enabled.")
            if self.guided_mode:
                self._speak("I will save future speech recordings.")
        elif mode == "save_prompts":
            self.save_prompts = True
            self.logger.enable_session_log = True
            self._print("[VOICE] Prompt and response logging enabled.")
            if self.guided_mode:
                self._speak("I will log prompts and responses.")
        self._print_status()

    def _handle_confirm_voice(self, tokens: List[str]):
        yes_words = self._get_yes_variants()
        no_words = self._get_no_variants()
        head = tokens[0]

        if self._fuzzy_in(head, yes_words):
            self._apply_confirm_decision(True)
            return

        if self._fuzzy_in(head, no_words):
            self._apply_confirm_decision(False)
            return

        if self.guided_mode:
            self._speak("Please say yes or no.")

    # --- Voice-command dispatcher ---------------------------------------

    def _handle_wake_command(self, tokens: List[str]):
        try:
            if not tokens:
                return
            cmd = " ".join(tokens)
            head = tokens[0]

            # Mode change
            if head == "mode" and len(tokens) >= 2:
                mode_word = tokens[1]
                guided_words = self._get_guided_variants()
                unguided_words = self._get_unguided_variants()
                if self._fuzzy_in(mode_word, guided_words):
                    self.guided_mode = True
                    self._print("[VOICE] Mode set to Guided.")
                    if self.guided_mode:
                        self._speak("Guided mode enabled.")
                elif self._fuzzy_in(mode_word, unguided_words):
                    self.guided_mode = False
                    self._print("[VOICE] Mode set to Unguided.")
                    self._speak("Unguided mode enabled.")
                else:
                    self._print("[VOICE] Could not understand mode setting. Say 'mode guided' or 'mode unguided'.")
                    if self.guided_mode:
                        self._speak("I did not catch guided or unguided.")
                return

            # Settings (voice)
            if head == "settings":
                self._print_settings()
                if self.guided_mode:
                    self._speak("Here are your current settings.")
                return

            # Prompt editing: start
            if head == "prompt":
                self.prompt_capture = True
                self._print("[VOICE] Prompt editing enabled. Speak to add text; say 'done' when finished.")
                if self.guided_mode:
                    self._speak("Prompt editing enabled. Begin speaking your prompt, and say done when finished.")
                return

            # Prompt editing: done / enter (spoken)
            if head in {"done", "enter"}:
                if self.prompt_capture:
                    self.prompt_capture = False
                    self._print("[VOICE] Prompt editing finished.")
                    if self.guided_mode:
                        self._speak("Prompt editing finished.")
                else:
                    self._print("[VOICE] No prompt is currently being edited.")
                return

            # Reasoning
            if head == "reasoning" and len(tokens) >= 2:
                level = tokens[1]
                levels = ["none", "low", "medium", "high"]
                matched = None
                for lv in levels:
                    if self._fuzzy_in(level, [lv]):
                        matched = lv
                        break
                if matched:
                    self.current_reasoning = matched
                    self._print(f"[VOICE] Reasoning set to: {matched}")
                    if self.guided_mode:
                        self._speak(f"Reasoning set to {matched}.")
                else:
                    self._print("[VOICE] Could not match reasoning level.")
                    if self.guided_mode:
                        self._speak("I could not match that reasoning level.")
                return

            # Model
            if head == "model":
                if len(tokens) >= 2:
                    tail = " ".join(tokens[1:])
                    self._handle_model_voice(tail)
                return

            # Buffer control: buffer clear / session / last
            if head == "buffer":
                if len(tokens) == 1:
                    self._print(
                        f"[VOICE] Buffer mode: {self.buffer_mode} "
                        f"(anchor={self.buffer_anchor_index}, last_action={self.buffer_last_action_index})"
                    )
                    return
                sub = tokens[1]
                if sub == "clear":
                    self.buffer_mode = "anchor"
                    self.buffer_anchor_index = len(self.terminal_buffer)
                    self._print("[VOICE] Buffer anchor set; future shell commands will use history after this point.")
                    if self.guided_mode:
                        self._speak("Buffer cleared for shell commands. I will use only new history.")
                elif sub == "session":
                    self.buffer_mode = "session"
                    self._print("[VOICE] Buffer mode set to session (full history for this run).")
                    if self.guided_mode:
                        self._speak("Using the full session history for shell commands.")
                elif sub == "last":
                    self.buffer_mode = "last"
                    self._print("[VOICE] Buffer mode set to last (since last Codex output or executed command).")
                    if self.guided_mode:
                        self._speak("Using only history since the last response or executed command.")
                else:
                    self._print("[VOICE] Unknown buffer setting. Say 'buffer clear', 'buffer session', or 'buffer last'.")
                    if self.guided_mode:
                        self._speak("Say buffer clear, buffer session, or buffer last.")
                return

            # History preview
            if head == "history":
                self._print_history_preview()
                if self.guided_mode:
                    self._speak("Showing a preview of the buffer I will send for shell commands.")
                return

            # Shell command suggestion (OS-level)
            if head == "shell" and len(tokens) >= 2 and tokens[1] == "command":
                self._print("[VOICE] Shell command: sending buffer and prompt to Codex for next shell command.")
                if self.guided_mode:
                    self._speak("Analyzing your shell history and prompt to suggest the next shell command.")
                self._voice_shell_request()
                return

            # Internal voice-command suggestion
            if head == "voice" and len(tokens) >= 2 and tokens[1] == "command":
                self._print("[VOICE] Voice command: asking Codex for an internal v- or voice command.")
                if self.guided_mode:
                    self._speak("Analyzing your context to suggest an internal voice command.")
                self._voice_app_command_request()
                return

            # Backwards compat alias: "voice shell" -> shell command
            if head == "voice" and len(tokens) >= 2 and tokens[1] == "shell":
                self._print("[VOICE] (Alias) Shell command: sending buffer and prompt to Codex.")
                if self.guided_mode:
                    self._speak("Analyzing your shell history and prompt to suggest the next shell command.")
                self._voice_shell_request()
                return

            # Execute pending command (voice)
            if head in {"execute", "exec"}:
                if not self.pending_shell_command:
                    self._print("[VOICE] No pending command to execute.")
                    if self.guided_mode:
                        self._speak("There is no pending command.")
                    return
                self.confirm_mode = "exec"
                self._print(f"[CONFIRM] Execute: {self.pending_shell_command}")
                if self.guided_mode:
                    if self.pending_shell_explanation:
                        self._speak(
                            f"{self.pending_shell_explanation}. "
                            "Do you want me to run this command? Say yes or no."
                        )
                    else:
                        self._speak("Do you want me to run this command? Say yes or no.")
                return

            # Repeat prompt
            if head == "repeat":
                if not self.prompt_text:
                    self._print("[VOICE] No prompt to repeat.")
                    if self.guided_mode:
                        self._speak("There is no current prompt.")
                    return
                self._print(f"[PROMPT] {self.prompt_text}")
                self._speak(self.prompt_text)
                return

            # Respond (explanation or last response)
            if self._fuzzy_in(head, ["respond", "response", "reply"]):
                expl = self.pending_shell_explanation or self.last_codex_response
                if not expl:
                    self._print("[VOICE] No response available to read.")
                    if self.guided_mode:
                        self._speak("There is no response yet.")
                    return
                self._print(f"[RESP] {expl}")
                self._speak(expl)
                return

            # Help
            if head == "help":
                self._speak_help()
                return

            # Exit (voice)
            if head == "exit":
                self.confirm_mode = "exit"
                self._print("[CONFIRM] Exit voice shell? Say yes or no.")
                self._speak("Do you want to exit the voice shell? Say yes or no.")
                return

            # Stop listening
            if head == "stop" and len(tokens) >= 2 and tokens[1] == "listening":
                self._listening = False
                self._print("[VOICE] Stop listening requested.")
                if self.guided_mode:
                    self._speak("Stopping listening.")
                return

            # Start listening
            if head == "listen" or (head == "start" and len(tokens) >= 2 and tokens[1] == "listening"):
                self._init_stt()
                if self.guided_mode:
                    self._speak("Listening enabled.")
                return

            # Speed control
            if head == "speed":
                if len(tokens) == 1:
                    self._print(f"[VOICE] Current TTS speed: {self.tts_speed:.2f}x")
                    if self.guided_mode:
                        self._speak(f"My speaking speed is {self.tts_speed:.2f} times normal.")
                    return
                arg = tokens[1]
                if arg in {"increase", "up", "faster"}:
                    self.tts_speed = min(TTS_SPEED_MAX, self.tts_speed + TTS_SPEED_STEP)
                elif arg in {"decrease", "down", "slower"}:
                    self.tts_speed = max(TTS_SPEED_MIN, self.tts_speed - TTS_SPEED_STEP)
                else:
                    try:
                        val = float(arg)
                        self.tts_speed = max(TTS_SPEED_MIN, min(TTS_SPEED_MAX, val))
                    except Exception:
                        self._print("[VOICE] Could not parse speed; use a number like 1.0, 1.2 or say 'speed increase'.")
                        if self.guided_mode:
                            self._speak("I did not understand that speed value.")
                        return
                self._print(f"[VOICE] TTS speed set to {self.tts_speed:.2f}x")
                if self.guided_mode:
                    self._speak(f"Speaking speed set to {self.tts_speed:.2f} times normal.")
                return

            # Color / style via voice
            if head in {"color", "style", "ui"} and len(tokens) >= 2:
                word = tokens[1]
                if word in {"on", "enable", "enabled", "fancy"}:
                    if colorama_init and Fore and Style:
                        if not self.use_color:
                            colorama_init()
                        self.use_color = True
                        self._print("[VOICE] Fancy colored output enabled.")
                        if self.guided_mode:
                            self._speak("Fancy colored output enabled.")
                    else:
                        self._print("[VOICE] Colorama not available; cannot enable fancy mode.")
                elif word in {"off", "disable", "disabled", "plain"}:
                    self.use_color = False
                    self._print("[VOICE] Fancy colored output disabled.")
                    if self.guided_mode:
                        self._speak("Fancy colored output disabled.")
                return

            # Debug logging (voice)
            if head == "debug" and len(tokens) >= 2:
                word = tokens[1]
                if word in {"on", "enable"}:
                    self.debug_enabled = True
                    self.logger.enable_debug = True
                    self._print("[VOICE] Debug logging enabled.")
                    if self.guided_mode:
                        self._speak("Debug logging is on.")
                elif word in {"off", "disable"}:
                    self.debug_enabled = False
                    self.logger.enable_debug = False
                    self._print("[VOICE] Debug logging disabled.")
                    if self.guided_mode:
                        self._speak("Debug logging is off.")
                return

            # Save recordings (voice, confirm)
            if head == "save" and len(tokens) >= 2 and tokens[1] == "recordings":
                state = "currently saving" if self.save_recordings else "currently not saving"
                self.confirm_mode = "save_recordings"
                msg = f"Save recordings is {state}. Would you like to enable saving recordings? Say yes or no."
                self._print(f"[CONFIRM] {msg}")
                if self.guided_mode:
                    self._speak(msg)
                return

            # Log prompts/responses (voice, confirm)
            if head == "log":
                state = (
                    "currently logging prompts and responses"
                    if self.save_prompts
                    else "currently not logging prompts and responses"
                )
                self.confirm_mode = "save_prompts"
                msg = (
                    f"Logging is {state}. "
                    "Would you like to enable logging prompts and responses? Say yes or no."
                )
                self._print(f"[CONFIRM] {msg}")
                if self.guided_mode:
                    self._speak(msg)
                return

            # Save settings (voice, confirm)
            if head == "save" and len(tokens) == 1:
                self.confirm_mode = "save_settings"
                msg = "Save current settings so they load automatically next time? Say yes or no."
                self._print(f"[CONFIRM] {msg}")
                if self.guided_mode:
                    self._speak(msg)
                return

            # Rename assistant
            if head == "rename" and len(tokens) >= 2:
                new_name = " ".join(tokens[1:]).strip()
                if not new_name:
                    self._print("[VOICE] Please provide a name to rename to.")
                    return
                self.assistant_name = new_name
                self.profile.add_aliases("wake", [new_name])
                self.profile.save()
                self._print(f"[VOICE] Assistant renamed to: {self.assistant_name}")
                if self.guided_mode:
                    self._speak(f"You can now call me {self.assistant_name}.")
                return

            # Run generic prompt
            if head == "run":
                if not self.prompt_text:
                    self._print("[VOICE] No prompt to run.")
                    if self.guided_mode:
                        self._speak("There is no prompt to run.")
                    return
                self._print("[VOICE] Running prompt via Codex.")
                if self.guided_mode:
                    self._speak("Running your prompt.")
                self.pending_shell_explanation = None
                self._run_codex_prompt(self.prompt_text)
                return

            # Self-directed tests
            if head == "test" and len(tokens) >= 2:
                mode_word = tokens[1]
                if mode_word in {"voice", "shell", "both"}:
                    self._run_self_test(mode_word)
                    return

            if head == "self" and len(tokens) >= 2 and tokens[1] in {"directed", "direct"}:
                self._run_self_test("both")
                return

            # Start listening: handled above

            # Fallback
            self._print(f"[VOICE] Unknown '{self.assistant_name.lower()}' command: {cmd}")
            if self.guided_mode:
                self._speak("I did not recognize that command.")
        finally:
            self._print_status()

    def _handle_model_voice(self, tail: str):
        models = ["gpt-5", "gpt-5-codex", "gpt-4.1", "gpt-4o"]
        idx = models.index(self.current_model) if self.current_model in models else 0
        tokens = tail.split()
        head = tokens[0] if tokens else ""

        if head in {"next", "forward"}:
            idx = (idx + 1) % len(models)
            self.current_model = models[idx]
            self._print(f"[VOICE] Model set to: {self.current_model}")
            if self.guided_mode:
                self._speak(f"Model set to {self.current_model}.")
            return

        if head in {"previous", "prev", "back"}:
            idx = (idx - 1) % len(models)
            self.current_model = models[idx]
            self._print(f"[VOICE] Model set to: {self.current_model}")
            if self.guided_mode:
                self._speak(f"Model set to {self.current_model}.")
            return

        name = tail.strip()
        if name in models:
            self.current_model = name
            self._print(f"[VOICE] Model set to: {self.current_model}")
            if self.guided_mode:
                self._speak(f"Model set to {self.current_model}.")
        else:
            self._print(f"[VOICE] Unknown model: {name}")

    # --- TTS helpers -----------------------------------------------------

    def _speak(self, text: str):
        if not self._tts or not text:
            return

        cleaned_lines = []
        for line in text.splitlines():
            ls = line.strip()
            if not ls:
                continue
            if ls.lower().startswith("model:"):
                continue
            if ls.lower().startswith("reasoning:"):
                continue
            if ls.startswith("- ") or ls.startswith("* "):
                ls = ls[2:]
            cleaned_lines.append(ls)

        if not cleaned_lines:
            return

        speak_text = " ".join(cleaned_lines)

        with self._tts_lock:
            try:
                speaker = None
                if self._tts_speakers and 0 <= self.voice_index < len(self._tts_speakers):
                    speaker = self._tts_speakers[self.voice_index]

                sr = getattr(self._tts, "output_sample_rate", None)
                if sr is None:
                    synth = getattr(self._tts, "synthesizer", None)
                    sr = getattr(synth, "output_sample_rate", 22050)

                wav = self._tts.tts(text=speak_text, speaker=speaker)

                if self.save_recordings and sf is not None:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    out_path = self.recordings_dir / f"tts_{ts}.wav"
                    try:
                        sf.write(str(out_path), wav, sr)
                        self._print(f"[VOICE] Saved speech to {out_path}", buffer=False)
                    except Exception as e:
                        self._print(f"[VOICE] Failed to save TTS audio: {e}", buffer=False)

                if sd is not None:
                    try:
                        sr_play = int(sr * self.tts_speed)
                        if sr_play < 8000:
                            sr_play = 8000
                        sd.play(wav, samplerate=sr_play)
                        sd.wait()
                    except Exception as e:
                        self._print(f"[VOICE] Audio playback error: {e}", buffer=False)
                else:
                    self._print("[VOICE] sounddevice not available; TTS audio not played.", buffer=False)

            except Exception as e:
                self._print(f"[VOICE] TTS error: {e}", buffer=False)

    def _speak_help(self):
        msg = (
            f"{self.assistant_name} voice help. "
            f"Say '{self.assistant_name} prompt' to dictate or edit a prompt, then speak your text. "
            f"Say '{self.assistant_name} done' to stop editing, or press Enter on an empty line. "
            f"Say '{self.assistant_name} run' to submit the prompt. "
            f"Say '{self.assistant_name} shell command' to analyze recent shell history and your prompt "
            f"and propose a single shell command plus explanation. "
            f"Say '{self.assistant_name} voice command' to get an internal control command like a v- command or voice command. "
            f"Say '{self.assistant_name} execute' to run the proposed command, after you say yes at the confirmation prompt. "
            f"Say '{self.assistant_name} respond' to hear the explanation or last response. "
            f"Say '{self.assistant_name} repeat' to hear and print your current prompt. "
            f"Say '{self.assistant_name} history' to preview the buffer slice I will send. "
            f"Say '{self.assistant_name} buffer clear', 'buffer session', or 'buffer last' to control "
            f"which part of history I use. "
            f"Say '{self.assistant_name} mode guided' or 'mode unguided' to control how talkative I am. "
            f"Say '{self.assistant_name} speed increase' or 'speed 1 point 2' to adjust my speaking rate. "
            f"Say '{self.assistant_name} save recordings' or '{self.assistant_name} log' to configure "
            f"saving and logging with confirmation. "
            f"Say '{self.assistant_name} save' to save all settings to disk, and '{self.assistant_name} exit' "
            f"to exit this voice shell. "
            f"You can also say '{self.assistant_name} test voice', '{self.assistant_name} test shell', "
            f"or '{self.assistant_name} test both' to run a self-test."
        )
        self._speak(msg)

    # --- Codex integration -----------------------------------------------

    def _build_codex_command(self, prompt: str) -> List[str]:
        return [
            "codex",
            "exec",
            "--skip-git-repo-check",
            "--model",
            self.current_model,
            "--config",
            f'model_reasoning_effort="{self.current_reasoning}"',
            prompt,
        ]

    def _run_codex_prompt(self, prompt: str):
        self.last_codex_prompt = prompt
        cmd = self._build_codex_command(prompt)
        self.logger.session(f"PROMPT: {prompt}")
        self._print()
        self._print("=" * 80)
        self._print("New Codex run:")
        self._print(f"Model: {self.current_model}, Reasoning: {self.current_reasoning}")
        self._print(f"Prompt:\n{prompt}")
        self._print("Command:")
        self._print(" ".join(shlex.quote(c) for c in cmd))

        self._print("┌────────────────────────────────────────┐")
        self._print("│  Running Codex, please wait...        │")
        self._print("└────────────────────────────────────────┘")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError:
            msg = (
                "Error: 'codex' CLI not found. "
                "Install with 'npm install -g @openai/codex' and run 'codex login'."
            )
            self._print(msg)
            self.last_codex_response = msg
            return
        except Exception as e:
            msg = f"Error running codex: {e}"
            self._print(msg)
            self.last_codex_response = msg
            return

        if result.returncode != 0:
            out = f"Error (exit code {result.returncode}):\n{result.stderr or '(no stderr)'}"
        else:
            out = result.stdout or "(no output)"

        self._print("Output:")
        self._print(out)
        self.last_codex_response = out
        self.logger.session(f"RESPONSE: {out}")

        self.buffer_last_action_index = len(self.terminal_buffer)

        if self.guided_mode:
            self._speak("Response ready.")

        self._print_status()

    def _get_history_for_shell(self) -> str:
        if not self.terminal_buffer:
            return ""
        if self.buffer_mode == "session":
            start = 0
        elif self.buffer_mode == "anchor":
            start = self.buffer_anchor_index
        elif self.buffer_mode == "last":
            start = self.buffer_last_action_index
        else:
            start = 0
        if start < 0 or start > len(self.terminal_buffer):
            start = 0
        lines = self.terminal_buffer[start:]
        return "\n".join(lines[-400:])

    def _print_history_preview(self, max_lines: int = 40):
        history = self._get_history_for_shell()
        if not history.strip():
            self._print("[HISTORY] No buffered history for shell command (slice is empty).", buffer=False)
            return
        lines = history.splitlines()
        self._print("[HISTORY] Preview of buffer slice used for shell commands:", buffer=False)
        start_idx = max(0, len(lines) - max_lines)
        for i in range(start_idx, len(lines)):
            num = i + 1
            self._print(f"[H] {num:4d}: {lines[i]}", buffer=False)

    def _extract_cmd_expl_from_response(self) -> (Optional[str], Optional[str]):
        raw = self.last_codex_response or ""
        lines = [l.strip() for l in raw.splitlines() if l.strip()]

        cmd_line = None
        expl_line = None

        for l in lines:
            up = l.upper()
            if up.startswith("CMD:"):
                cmd_line = l[4:].strip()
            elif up.startswith("EXPL:"):
                expl_line = l[5:].strip()

        if cmd_line is None and lines:
            cmd_line = lines[0]
        if expl_line is None and len(lines) > 1:
            expl_line = lines[1]

        return cmd_line, expl_line

    def _voice_shell_request(self):
        history = self._get_history_for_shell()
        if not history and not self.prompt_text.strip():
            self._print("[SHELL] No history or prompt to analyze.")
            if self.guided_mode:
                self._speak("There is no buffer or prompt yet.")
            return

        user_req = self.prompt_text.strip() or "(none)"

        prompt = (
            "You are an expert Linux shell assistant.\n"
            "You will receive:\n"
            " - The user's high-level request (may be empty).\n"
            " - Recent terminal history (commands and outputs).\n\n"
            "Your job:\n"
            " - Decide the single best next shell command to run.\n"
            " - Explain briefly what it does.\n\n"
            "IMPORTANT OUTPUT FORMAT:\n"
            "Return EXACTLY TWO non-empty lines, and nothing else:\n"
            "  LINE 1: CMD: <the exact shell command to run>\n"
            "  LINE 2: EXPL: <a short natural language explanation, <= 25 words>\n"
            "No backticks, no extra commentary, no bullet points.\n\n"
            "USER HIGH-LEVEL REQUEST:\n"
            f"{user_req}\n\n"
            f"BUFFER MODE: {self.buffer_mode}\n"
            "=== TERMINAL HISTORY START ===\n"
            f"{history}\n"
            "=== TERMINAL HISTORY END ===\n"
        )

        self.pending_shell_explanation = None
        self._run_codex_prompt(prompt)

        cmd_line, expl_line = self._extract_cmd_expl_from_response()

        if cmd_line:
            self.pending_shell_command = cmd_line
            self.pending_shell_explanation = expl_line or ""
            self._print(f"[SHELL] Proposed shell command: {self.pending_shell_command}")
            if self.pending_shell_explanation:
                self._print(f"[SHELL] Explanation: {self.pending_shell_explanation}")
            if self.guided_mode:
                if self.pending_shell_explanation:
                    self._speak(
                        f"{self.pending_shell_explanation}. "
                        "You can say execute to run it, or respond to hear this again."
                    )
                else:
                    self._speak("I have proposed a shell command. Say execute to run it, or respond to hear the response.")
        else:
            self.pending_shell_command = None
            self.pending_shell_explanation = None
            self._print("[SHELL] Could not extract a single command from Codex response.")
            if self.guided_mode:
                self._speak("I could not extract a single command from the response.")
        self._print_status()

    def _voice_app_command_request(self):
        """Ask Codex for an internal 'v-' or '<AssistantName> ...' command."""
        history = self._get_history_for_shell()
        user_req = self.prompt_text.strip() or "(none)"

        prompt = (
            "You are the control logic for a voice-enabled shell application called Alice Shell.\n"
            "The app supports two kinds of internal commands:\n"
            "  1) Typed commands starting with 'v-' such as:\n"
            "     v-help, v-settings, v-prompt, v-command, v-voicecmd, v-history, v-buffer clear,\n"
            "     v-guided-on, v-guided-off, v-speed 1.2, v-respond, v-repeat, v-recordings, v-logprompts,\n"
            "     v-debug, v-save, v-exec, v-exit, v-listen.\n"
            "  2) Voice commands starting with the assistant name, for example:\n"
            "     Alice prompt, Alice done, Alice run, Alice shell command,\n"
            "     Alice voice command, Alice execute, Alice respond, Alice repeat,\n"
            "     Alice history, Alice buffer last, Alice mode guided, Alice speed increase,\n"
            "     Alice test voice, Alice test shell, Alice test both.\n\n"
            "Your job:\n"
            "  - Look at the user's high-level request and the recent shell history.\n"
            "  - Choose the single most helpful internal control command to invoke next.\n"
            "  - Prefer 'v-' commands when it is something the user might type, or\n"
            "    an 'Alice ...' voice command when the user is working by voice.\n"
            "  - Do NOT return plain shell commands like 'ls', 'git status', etc.\n\n"
            "IMPORTANT OUTPUT FORMAT:\n"
            "Return EXACTLY TWO non-empty lines, and nothing else:\n"
            "  LINE 1: CMD: <the internal command: either v-... or 'Alice ...'>\n"
            "  LINE 2: EXPL: <a short natural language explanation, <= 25 words>\n"
            "No backticks, no extra commentary.\n\n"
            "USER HIGH-LEVEL REQUEST:\n"
            f"{user_req}\n\n"
            "=== RECENT SHELL HISTORY (may be empty) ===\n"
            f"{history}\n"
            "=== END HISTORY ===\n"
        )

        self.pending_shell_explanation = None
        self._run_codex_prompt(prompt)

        cmd_line, expl_line = self._extract_cmd_expl_from_response()

        if cmd_line:
            self.pending_shell_command = cmd_line
            self.pending_shell_explanation = expl_line or ""
            self._print(f"[SHELL] Proposed voice command: {self.pending_shell_command}")
            if self.pending_shell_explanation:
                self._print(f"[SHELL] Explanation: {self.pending_shell_explanation}")
            if self.guided_mode:
                if self.pending_shell_explanation:
                    self._speak(
                        f"{self.pending_shell_explanation}. "
                        "You can say execute to run this command, or respond to hear this again."
                    )
                else:
                    self._speak("I have proposed an internal command. Say execute to run it, or respond to hear the response.")
        else:
            self.pending_shell_command = None
            self.pending_shell_explanation = None
            self._print("[SHELL] Could not extract an internal command from Codex response.")
            if self.guided_mode:
                self._speak("I could not extract an internal command from the response.")
        self._print_status()

    # --- Self-test -------------------------------------------------------

    def _run_self_test(self, mode: str):
        """
        Experimental self-test.
        mode: "voice", "shell", or "both"
        Does NOT auto-run any OS shell commands.
        """
        mode = mode.lower()
        self._print(f"[TEST] Starting self-test mode ({mode}).", buffer=False)
        if self.guided_mode:
            self._speak(
                f"Starting self test for {mode} systems. "
                "I will not run any operating system commands without your confirmation."
            )

        def record(msg: str):
            self._print(f"[TEST] {msg}", buffer=False)
            self.logger.debug(f"[TEST] {msg}")

        # Voice command logic test (no actual microphone loopback)
        if mode in {"voice", "both"}:
            record("Testing voice command logic (internal handlers only).")
            try:
                self._handle_wake_command(["settings"])
                record("Voice 'settings' handled without exception.")
            except Exception as e:
                record(f"ERROR in voice 'settings' handler: {e}")

            try:
                self._handle_wake_command(["prompt"])
                record("Voice 'prompt' handler executed (prompt editing started).")
                # Simulate dictation fragment
                self.prompt_text += " self-test prompt fragment"
                self._handle_wake_command(["done"])
                record("Voice 'done' handler executed (prompt editing stopped).")
            except Exception as e:
                record(f"ERROR in voice prompt/edit handlers: {e}")

            try:
                self._handle_wake_command(["mode", "guided"])
                self._handle_wake_command(["mode", "unguided"])
                record("Voice 'mode guided/unguided' handlers executed.")
            except Exception as e:
                record(f"ERROR in voice mode handlers: {e}")

            try:
                self._handle_wake_command(["speed", "1"])
                self._handle_wake_command(["speed", "increase"])
                record("Voice 'speed' handlers executed.")
            except Exception as e:
                record(f"ERROR in voice speed handlers: {e}")

        # Shell command suggestion test (no OS execution)
        if mode in {"shell", "both"}:
            record("Testing shell-command suggestion logic (no OS execution).")
            try:
                # Seed some fake history
                self.terminal_buffer.append("echo 'hello from self-test'")
                self.terminal_buffer.append("ls -la")
                old_prompt = self.prompt_text
                self.prompt_text = "Summarize and suggest next maintenance command."
                self._voice_shell_request()
                record("Shell command suggestion completed.")
                self.prompt_text = old_prompt
                # Do NOT call execute; no OS commands run.
            except Exception as e:
                record(f"ERROR in shell command suggestion: {e}")

        # Basic health check
        stt_ok = (sd is not None and self._vosk_model is not None and self._recognizer is not None)
        tts_ok = (self._tts is not None and sd is not None)
        log_ok = (self.logger.enable_debug or self.logger.enable_session_log)

        record(f"STT available: {'YES' if stt_ok else 'NO'}")
        record(f"TTS available: {'YES' if tts_ok else 'NO'}")
        record(f"Logging enabled: {'YES' if log_ok else 'NO'}")

        if self.guided_mode:
            summary_parts = ["Self test finished."]
            if not stt_ok:
                summary_parts.append("Speech recognition is not fully available.")
            if not tts_ok:
                summary_parts.append("Speech output is not fully available.")
            if not log_ok:
                summary_parts.append("Logging is currently limited.")
            if stt_ok and tts_ok:
                summary_parts.append("Core voice pipeline appears healthy.")
            self._speak(" ".join(summary_parts))

    # --- Shell command & typed control -----------------------------------

    def _run_shell_command(self, command: str):
        if not command.strip():
            return
        self._print(f"$ {command}")
        try:
            result = subprocess.run(command, shell=True, text=True, capture_output=True)
        except Exception as e:
            self._print(f"[SHELL] Error running command: {e}")
            return
        if result.stdout:
            for line in result.stdout.splitlines():
                self._print(line)
        if result.stderr:
            for line in result.stderr.splitlines():
                self._print(line)
        self.buffer_last_action_index = len(self.terminal_buffer)
        self._print_status()

    def _print_settings(self):
        self._print("=== Settings ===")
        self._print(f"Assistant name: {self.assistant_name}")
        self._print(f"Guided mode: {'ON' if self.guided_mode else 'OFF'}")
        self._print(f"Model: {self.current_model}")
        self._print(f"Reasoning: {self.current_reasoning}")
        self._print(f"Debug logging: {'ON' if self.debug_enabled else 'OFF'}")
        self._print(f"Save recordings: {'ON' if self.save_recordings else 'OFF'}")
        self._print(f"Log prompts/responses: {'ON' if self.save_prompts else 'OFF'}")
        self._print(f"Fancy output: {'ON' if self.use_color else 'OFF'}")
        self._print(f"TTS speed: {self.tts_speed:.2f}x")
        self._print(
            f"Buffer mode: {self.buffer_mode} "
            f"(anchor={self.buffer_anchor_index}, last_action={self.buffer_last_action_index})"
        )
        if self._tts_speakers:
            current = (
                self._tts_speakers[self.voice_index]
                if 0 <= self.voice_index < len(self._tts_speakers)
                else "default"
            )
            self._print(f"Voice index: {self.voice_index+1} / {len(self._tts_speakers)} ({current})")
        else:
            self._print("Voice: default / single-speaker")
        self._print_status()

    def _handle_typed_voice_command(self, line: str):
        cmd = line[2:].strip()
        if not cmd:
            return
        parts = cmd.split()
        key = parts[0]

        if key == "help":
            self._print_banner()
            return

        if key == "settings":
            self._print_settings()
            return

        if key == "guided-on":
            self.guided_mode = True
            self._print("[CMD] Guided mode enabled.")
            self._print_status()
            return

        if key == "guided-off":
            self.guided_mode = False
            self._print("[CMD] Guided mode disabled.")
            self._print_status()
            return

        if key == "debug":
            self.debug_enabled = not self.debug_enabled
            self.logger.enable_debug = self.debug_enabled
            self._print(f"[CMD] Debug logging {'enabled' if self.debug_enabled else 'disabled'}.")
            return

        if key == "saveprompts-on":
            self.save_prompts = True
            self.logger.enable_session_log = True
            self._print("[CMD] Prompt/response logging enabled.")
            return

        if key == "saveprompts-off":
            self.save_prompts = False
            self.logger.enable_session_log = False
            self._print("[CMD] Prompt/response logging disabled.")
            return

        if key == "recordspeech-on":
            self.save_recordings = True
            self._print("[CMD] Recording of speech responses enabled.")
            return

        if key == "recordspeech-off":
            self.save_recordings = False
            self._print("[CMD] Recording of speech responses disabled.")
            return

        if key == "recordings":
            self.save_recordings = not self.save_recordings
            self._print(f"[CMD] Save recordings: {'ON' if self.save_recordings else 'OFF'}.")
            return

        if key == "logprompts":
            self.save_prompts = not self.save_prompts
            self.logger.enable_session_log = self.save_prompts
            self._print(f"[CMD] Log prompts/responses: {'ON' if self.save_prompts else 'OFF'}.")
            return

        if key == "voice" and len(parts) >= 2:
            try:
                idx = int(parts[1]) - 1
                if idx < 0:
                    raise ValueError
                self.voice_index = idx
                self._print(f"[CMD] Voice index set to {idx+1}.")
            except Exception:
                self._print("[CMD] Usage: v-voice N (1-based index)")
            return

        if key == "rename" and len(parts) >= 2:
            new_name = " ".join(parts[1:]).strip().strip('"')
            if not new_name:
                self._print("[CMD] Usage: v-rename <name>")
                return
            self.assistant_name = new_name
            self.profile.add_aliases("wake", [new_name])
            self.profile.save()
            self._print(f"[CMD] Assistant renamed to: {self.assistant_name}")
            return

        if key == "model" and len(parts) >= 2:
            model = parts[1]
            self.current_model = model
            self._print(f"[CMD] Model set to: {model}")
            self._print_status()
            return

        if key == "reasoning" and len(parts) >= 2:
            self.current_reasoning = parts[1]
            self._print(f"[CMD] Reasoning set to: {self.current_reasoning}")
            self._print_status()
            return

        if key in {"shell", "command"}:
            self._voice_shell_request()
            return

        if key in {"voicecmd", "voice-command", "voicecommand"}:
            self._voice_app_command_request()
            return

        if key == "prompt":
            self.prompt_capture = True
            self._print("[CMD] Prompt editing enabled. Current prompt is:")
            self._print(self.prompt_text or "<empty>")
            try:
                typed = input("PROMPT> ")
            except EOFError:
                typed = ""
            if typed.strip():
                self.prompt_text = typed
                self._print(f"[CMD] Prompt set to: {self.prompt_text}")
            self._print_status()
            return

        if key == "exec":
            if not self.pending_shell_command:
                self._print("[CMD] No pending command.")
                return
            self._print(f"[CONFIRM] Execute: {self.pending_shell_command} (y/N)? ", end="", buffer=True)
            ans = input().strip().lower()
            if ans in {"y", "yes", "confirm"}:
                cmd_to_run = self.pending_shell_command.strip()
                self._print(f"[EXEC] Running: {cmd_to_run}")
                if cmd_to_run.startswith("v-"):
                    # Internal typed command
                    self._handle_typed_voice_command(cmd_to_run)
                else:
                    tokens = cmd_to_run.split()
                    if tokens and tokens[0].lower() == self.assistant_name.lower():
                        # Voice-style command
                        self._handle_wake_command(self._tokenize(" ".join(tokens[1:])))
                    else:
                        # Shell command
                        self._run_shell_command(cmd_to_run)
                self.pending_shell_command = None
                self.pending_shell_explanation = None
            else:
                self._print("[CONFIRM] Cancelled.")
            return

        if key == "clear":
            self.terminal_buffer.clear()
            self._print("[CMD] Buffer cleared.")
            self.buffer_anchor_index = len(self.terminal_buffer)
            self.buffer_last_action_index = len(self.terminal_buffer)
            self._print_status()
            return

        if key == "repeat":
            if not self.prompt_text:
                self._print("[CMD] No prompt to repeat.")
            else:
                self._print(f"[PROMPT] {self.prompt_text}")
                self._speak(self.prompt_text)
            return

        if key == "respond":
            expl = self.pending_shell_explanation or self.last_codex_response
            if not expl:
                self._print("[CMD] No response available.")
                return
            self._print(f"[RESP] {expl}")
            self._speak(expl)
            return

        if key == "fancy-on":
            if colorama_init and Fore and Style:
                if not self.use_color:
                    colorama_init()
                self.use_color = True
                self._print("[CMD] Fancy colored output enabled.")
            else:
                self._print("[CMD] Colorama not available; cannot enable fancy mode.")
            return

        if key == "fancy-off":
            self.use_color = False
            self._print("[CMD] Fancy colored output disabled.")
            return

        if key == "speed" and len(parts) >= 2:
            try:
                val = float(parts[1])
                self.tts_speed = max(TTS_SPEED_MIN, min(TTS_SPEED_MAX, val))
                self._print(f"[CMD] TTS speed set to {self.tts_speed:.2f}x")
                self._print_status()
            except Exception:
                self._print("[CMD] Usage: v-speed 1.0 (range 0.5 - 2.0)")
            return

        if key == "buffer":
            if len(parts) == 1:
                self._print(
                    f"[CMD] Buffer mode: {self.buffer_mode} "
                    f"(anchor={self.buffer_anchor_index}, last_action={self.buffer_last_action_index})"
                )
                return
            sub = parts[1]
            if sub == "clear":
                self.buffer_mode = "anchor"
                self.buffer_anchor_index = len(self.terminal_buffer)
                self._print("[CMD] Buffer anchor set; future shell commands will use history after this point.")
            elif sub == "session":
                self.buffer_mode = "session"
                self._print("[CMD] Buffer mode set to session (full history for this run).")
            elif sub == "last":
                self.buffer_mode = "last"
                self._print("[CMD] Buffer mode set to last (since last Codex output or executed command).")
            else:
                self._print("[CMD] Usage: v-buffer clear|session|last")
            self._print_status()
            return

        if key == "history":
            self._print_history_preview()
            return

        if key == "listen":
            self._init_stt()
            return

        if key == "save":
            self._print("[CONFIRM] Save current settings to disk so they load next time? (y/N)? ", end="", buffer=True)
            ans = input().strip().lower()
            if ans in {"y", "yes", "confirm"}:
                self._save_settings()
            else:
                self._print("[CONFIRM] Settings save cancelled.")
            return

        if key == "exit":
            self._print("[CONFIRM] Exit voice shell? (y/N)? ", end="", buffer=True)
            ans = input().strip().lower()
            if ans in {"y", "yes", "confirm"}:
                self._exit_requested = True
                self._listening = False
                self._print("[SHELL] Exit requested.")
            else:
                self._print("[CONFIRM] Exit cancelled.")
            return

        self._print(f"[CMD] Unknown v- command: {cmd}")

    # --- Main loop -------------------------------------------------------

    def run(self):
        self._print_banner()
        announce = f"{self.assistant_name.lower()}-shell listening"
        self._print(f"[VOICE] {announce}")
        self._speak(announce)

        while True:
            if self._exit_requested:
                self._print("[SHELL] Exiting voice shell.")
                break

            try:
                line = input("$ ")
            except EOFError:
                self._print()
                self._print("[SHELL] EOF received, exiting.")
                break
            except KeyboardInterrupt:
                self._print()
                self._print("[SHELL] KeyboardInterrupt, exiting.")
                break

            stripped = line.strip()

            # Typed confirmation when confirm_mode is active
            if self.confirm_mode and stripped.lower() in {"yes", "y", "confirm", "no", "n", "cancel"}:
                accepted = stripped.lower() in {"yes", "y", "confirm"}
                self._apply_confirm_decision(accepted)
                continue

            if self._exit_requested:
                self._print("[SHELL] Exiting voice shell.")
                break

            # Empty line: if in prompt editing, treat as "done"
            if not stripped:
                if self.prompt_capture:
                    self.prompt_capture = False
                    self._print("[SHELL] Prompt editing finished (Enter).")
                    if self.guided_mode:
                        self._speak("Prompt editing finished.")
                    self._print_status()
                continue

            if stripped.startswith("v-"):
                self._handle_typed_voice_command(stripped)
                continue

            if stripped in {"exit", "quit"}:
                self._print("[CONFIRM] Exit voice shell? (y/N)? ", end="", buffer=True)
                ans = input().strip().lower()
                if ans in {"y", "yes", "confirm"}:
                    self._print("[SHELL] Goodbye.")
                    break
                else:
                    self._print("[CONFIRM] Exit cancelled.")
                    continue

            if stripped == "clear":
                os.system("clear")
                self.terminal_buffer.clear()
                self._print("[SHELL] Screen and buffer cleared.")
                self.buffer_anchor_index = len(self.terminal_buffer)
                self.buffer_last_action_index = len(self.terminal_buffer)
                self._print_status()
                continue

            self._run_shell_command(stripped)

        self._listening = False


def main():
    args = sys.argv[1:]
    shell = CodexVoiceShell(args)
    shell.run()


if __name__ == "__main__":
    main()

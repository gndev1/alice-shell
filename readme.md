<img src="alice-shell.png" alt="alice-shell logo" width="40%"/>
# alice-shell

Voice-enabled shell + GUI assistant for the Codex / GPT CLI.

- `alice-shell.py` – the main **voice shell** that wraps your terminal:
  - listens for voice commands,
  - can send your terminal buffer + prompt to Codex to propose **shell commands**,
  - can also ask Codex for **internal control commands** (e.g. `v-` commands),
  - only executes commands **after confirmation**.
- `alice.py` – a simpler **GUI frontend**:
  - good for testing microphones and text-to-speech voices,
  - and recalibrating speech recognition / wake-word behavior.

Both share the same calibration/profile files so you can tune recognition in the GUI, then benefit from it in the shell.

---

## Table of Contents

- [Features](#features)
  - [alice-shell.py (voice shell)](#alice-shellpy-voice-shell)
  - [alice.py (GUI)](#alicepy-gui)
- [Architecture & Files](#architecture--files)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage: alice-shell.py](#usage-alice-shellpy)
  - [Launch](#launch)
  - [Normal shell vs `v-` commands](#normal-shell-vs-v-commands)
  - [Voice commands](#voice-commands)
  - [Shell command suggestions](#shell-command-suggestions)
  - [Voice-command suggestions](#voice-command-suggestions)
  - [Buffer modes & history](#buffer-modes--history)
  - [Settings, saving & profiles](#settings-saving--profiles)
  - [Self-test / self-directed mode](#self-test--self-directed-mode)
- [Usage: alice.py (GUI)](#usage-alicepy-gui)
- [Logging, recordings & data](#logging-recordings--data)
- [License](#license)

---

## Features

### `alice-shell.py` (voice shell)

- **Full shell passthrough**  
  Type any normal shell command and it runs as usual. Alice just observes and logs.

- **Voice control**

  - Wake name (default `Alice`, customizable).
  - Vosk offline STT listens continuously in the background.
  - Coqui TTS speaks responses.

- **Smart command routing**

  - **Shell suggestions** – “Alice shell command”:
    - Sends your **terminal buffer + prompt** to Codex.
    - Codex returns:

      - `CMD: <shell command>`
      - `EXPL: <short explanation>`

    - Alice can read the **explanation** aloud.
    - `Alice execute` (or `v-exec`) runs the command **after confirmation**.

  - **Voice-command suggestions** – “Alice voice command”:
    - Asks Codex for the **next internal command**:
      - Either a **typed** command like `v-settings`, `v-guided-on`, `v-respond`, etc.
      - Or a **voice command** like `Alice prompt`, `Alice test both`, etc.
    - Again returns `CMD:` + `EXPL:`.
    - `Alice execute` or `v-exec` will:
      - Treat `v-*` as an internal typed command,
      - Treat `<AssistantName> ...` as a voice command,
      - Never send those to the OS shell.

- **Prompt editing (voice + keyboard)**

  - `Alice prompt` → enter **prompt editing mode**:
    - speak to append text,
    - or type & edit as normal,
    - you **do not** need to keep saying “Alice …” while dictating.
  - `Alice done` (or press Enter on an empty line) → exit prompt editing.
  - `Alice run` → send current prompt to Codex.
  - `Alice repeat` / `v-repeat` → print & read the current prompt aloud.

- **Buffer-aware context**

  - Uses a slice of the shell history as “context” for shell recommendations.
  - Configurable buffer modes: `session`, `anchor`, `last`.
  - `Alice buffer clear / session / last` or `v-buffer ...` to change it.
  - `Alice history` / `v-history` shows what will be sent.

- **Guided vs Unguided mode**

  - **Guided**: Alice explains what she’s doing, confirms actions, announces tests, etc.
  - **Unguided**: Minimal chatter, still functional.
  - Toggle with:

    - voice: `Alice mode guided` / `Alice mode unguided`
    - typed: `v-guided-on` / `v-guided-off`

  - Defaults to **Guided ON**.

- **Configurable TTS voice & speed**

  - Uses Coqui TTS multi-speaker model (`tts_models/en/vctk/vits`).
  - Voice index: `v-voice 2`, `v-voice 3`, etc.
  - Speed:

    - voice: `Alice speed`, `Alice speed increase`, `Alice speed 1.2`
    - typed: `v-speed 1.2`

  - Default: **speed 1.20x**.

- **Settings & persistence**

  - Current defaults:

        Assistant name: Alice
        Guided mode: ON
        Model: gpt-5
        Reasoning: medium
        Debug logging: ON
        Save recordings: ON
        Log prompts/responses: ON
        Fancy output: ON
        TTS speed: 1.20x
        Buffer mode: session

  - Save with:

    - typed: `v-save`
    - voice: `Alice save` (asks for yes/no)

  - Settings auto-load on next run (from `codex_shell_settings.json`).

- **Self-directed test mode (experimental)**

  - `Alice test voice` – exercises internal voice handlers.
  - `Alice test shell` – tests shell-suggestion logic without running OS commands.
  - `Alice test both` / `Alice self directed` – runs both tests.
  - Logs `[TEST]` lines and speaks a short summary in guided mode.

---

### `alice.py` (GUI)

Simpler but handy for **calibration & voice testing**:

- Tkinter GUI:

  - Model dropdown (e.g. `gpt-5`, `gpt-4o`, etc.).
  - Reasoning dropdown (`none`, `low`, `medium`, `high`).
  - Big Prompt + Response text areas.
  - “Speak response” button.

- Voice integration:

  - Same Vosk STT + Coqui TTS backend as the shell.
  - Assistant name field to change “Alice” to anything.
  - Voice dropdown to pick Coqui speaker.

- Calibration tools:

  - Guided calibration flow for wake word & core commands.
  - Per-word calibration list (words + aliases).
  - Saves your calibration profile as JSON shared by both GUI and shell.

Use `alice.py` to fine-tune recognition for your voice, then use `alice-shell.py` for serious CLI work.

---

## Architecture & Files

Example layout:

    alice-shell/
      alice-shell.py                # main voice shell
      alice.py                      # GUI frontend
      vosk-model-small-en-us-0.15/  # Vosk STT model (downloaded separately)
      recordings/                   # saved TTS audio (if enabled)
      logs/
        codex_shell_debug.log
        codex_shell_session.log
      codex_voice_profile.json      # calibration / alias profile
      codex_shell_settings.json     # saved shell settings
      README.md                     # this file

---

## Dependencies

**System:**

- Python 3.9+ (tested on Linux / Pop!_OS with 3.10)
- Node.js >= 16 (for Codex CLI)
- Working microphone + speakers/headphones

**Python packages (shell & GUI):**

    pip install sounddevice vosk TTS soundfile colorama

- `sounddevice` – audio input/output  
- `vosk` – offline speech-to-text  
- `TTS` – Coqui TTS (text-to-speech)  
- `soundfile` – saving wav files  
- `colorama` – colored CLI output

**Codex CLI:**

    npm install -g @openai/codex
    codex login

**Vosk model:**

Download a small English model (or your language) and place it as:

    vosk-model-small-en-us-0.15/

in the same directory as `alice-shell.py` and `alice.py`.

---

## Installation

1. Clone or copy the project directory containing `alice-shell.py` and `alice.py`.

2. Install Python dependencies:

       pip install sounddevice vosk TTS soundfile colorama

3. Install Codex CLI (requires Node >= 16):

       npm install -g @openai/codex
       codex login

4. Download Vosk model (`vosk-model-small-en-us-0.15`) and put it under the project root.

5. (Optional) Test Coqui TTS:

       python3 -c "from TTS.api import TTS; t=TTS('tts_models/en/vctk/vits'); t.tts_to_file('Hello from Alice shell', 'hello.wav')"

---

## Usage: `alice-shell.py`

### Launch

From the project folder:

    python3 alice-shell.py

Alice will announce something like:

    alice-shell listening

Startup options (all optional):

- `-guided` / `-unguided` – default guided mode
- `-logdebug` / `-nodebug` – debug logging
- `-recordresponsespeech` / `-norecordresponsespeech`
- `-saveprompts` / `-nosaveprompts`
- `-voice:N` – pick Coqui speaker by index (1-based)
- `-speed:X` – TTS speed (0.5 – 2.0)
- `-fancy` / `-nofancy` – colored output

Example:

    python3 alice-shell.py -guided -fancy -speed:1.2 -voice:2

---

### Normal shell vs `v-` commands

Normal commands (no prefix) go straight to your system shell:

    $ ls
    $ git status
    $ python3 script.py

Voice-shell commands from the keyboard use the `v-` prefix:

    $ v-help            # show voice-shell help
    $ v-settings        # show current settings
    $ v-prompt          # open prompt editor
    $ v-command         # ask Codex for next shell command
    $ v-voicecmd        # ask Codex for next internal voice command
    $ v-history         # view buffer slice
    $ v-buffer last     # use only history since last Codex action
    $ v-guided-on       # enable guided mode
    $ v-guided-off      # disable guided mode
    $ v-speed 1.2       # set TTS speed
    $ v-recordings      # toggle saving TTS audio
    $ v-logprompts      # toggle prompt/response logging
    $ v-debug           # toggle debug logging
    $ v-respond         # speak last explanation / response
    $ v-repeat          # repeat current prompt
    $ v-listen          # (re)start microphone listening
    $ v-save            # save settings (with confirm)
    $ v-exec            # execute last proposed command (v- or shell)
    $ v-exit            # exit the voice shell

---

### Voice commands

Use the assistant name (default `Alice`) as your wake word.

**Settings & name**

- `Alice settings` – read current settings.
- `Alice rename Samantha` – change assistant name.
- `Alice mode guided` / `Alice mode unguided`.
- `Alice debug on` / `Alice debug off`.
- `Alice save recordings` – turn saving TTS recordings on/off (yes/no).
- `Alice log` – turn prompt/response logging on/off (yes/no).
- `Alice save` – save settings to file (yes/no).

**Prompt editing**

- `Alice prompt` – enter prompt editing mode:
  - speak new text (it’s appended),
  - or type and edit.
- `Alice done` – stop prompt editing (or press Enter on empty line).
- `Alice run` – send prompt to Codex.
- `Alice repeat` – read and print the current prompt.

**Listening / TTS**

- `Alice listen` – (re)start microphone listening.
- `Alice stop listening` – stop microphone input.
- `Alice speed` – speak current speed.
- `Alice speed increase` / `Alice speed decrease`.
- `Alice speed 1.2` – set speed explicitly.

**Help & exit**

- `Alice help` – spoken help about commands.
- `Alice exit` – quit `alice-shell` (with yes/no confirmation).

---

### Shell command suggestions

To ask Codex what to run next:

1. Use your shell as usual.
2. Optionally build a prompt via `Alice prompt` or `v-prompt`.
3. Then:

   - Voice: `Alice shell command`
   - Typed: `v-command`

Alice will:

- Collect a slice of your shell history (based on buffer mode).
- Build a detailed prompt to Codex describing:

  - your high-level goal (from the prompt),
  - recent shell history,
  - and a strict output format:

        CMD: <shell command>
        EXPL: <short explanation>

- Save Codex’s raw output in the Response log.
- Parse `CMD:` and `EXPL:` and show, for example:

        [SHELL] Proposed shell command: git commit -am "Fix config"
        [SHELL] Explanation: Stage and commit modified files with message "Fix config".

Then you can:

- `Alice respond` / `v-respond` – to hear the explanation.
- `Alice execute` / `v-exec` – to run the command, with confirmation.

---

### Voice-command suggestions

To have Codex suggest an internal command (not a bash command):

- Voice: `Alice voice command`
- Typed: `v-voicecmd`

Codex is asked to return **either**:

- a `v-...` command (like `v-settings`, `v-buffer last`, `v-save`),
- or a voice command like `Alice test both`.

Alice will:

- Store it as the pending command plus explanation.
- In guided mode, speak the explanation.
- On `Alice execute` / `v-exec`:
  - If it’s `v-...` → run it as an internal typed command.
  - If it starts with the assistant name → run it as if spoken.
  - Never send these to the OS shell.

---

### Buffer modes & history

Controls which part of terminal history is sent as context for `Alice shell command` / `v-command`.

Modes:

- `session` – everything since `alice-shell.py` started (default).
- `anchor` – everything after a movable anchor.
- `last` – everything since the last Codex response or executed command.

Change with:

- Voice:

  - `Alice buffer clear` – set anchor to “now”; use only new history.
  - `Alice buffer session` – full session history.
  - `Alice buffer last` – only the last interaction.

- Typed:

  - `v-buffer clear`
  - `v-buffer session`
  - `v-buffer last`

Preview the buffer slice:

    v-history
    # or
    Alice history

---

### Settings, saving & profiles

View settings:

    v-settings
    # or
    Alice settings

Save settings to `codex_shell_settings.json`:

    v-save
    # or
    Alice save

Calibration and alias data (wake-word variants, “guided/unguided” words, yes/no variants, etc.) live in:

    codex_voice_profile.json

This profile is used by both the GUI and shell and is loaded automatically.

---

### Self-test / self-directed mode

Experimental self-test to check if major components work:

- `Alice test voice` – tests internal voice-command handling.
- `Alice test shell` – tests shell-command suggestion prompting (no actual OS execution).
- `Alice test both` / `Alice self directed` – runs both.

Test output looks like:

    [TEST] Testing voice command logic (internal handlers only).
    [TEST] Testing shell-command suggestion logic (no OS execution).
    [TEST] STT available: YES
    [TEST] TTS available: YES
    [TEST] Logging enabled: YES

In guided mode Alice will also speak a brief health summary at the end.

---

## Usage: `alice.py` (GUI)

Run:

    python3 alice.py

You’ll see:

- Model + reasoning dropdowns.
- Large Prompt text area.
- Large Response text area.
- A Listening toggle.
- Assistant name field (e.g. “Alice”, “Samantha”).
- Voice dropdown (Coqui speakers).
- “Speak response” button to read out the current Response.

Use cases:

- Test mic and Coqui voice before using the shell.
- Calibrate wake name and key command words (per-word calibration).
- Experiment with speakers and speeds.

Any calibration/profile data saved from the GUI is read by the shell on startup.

---

## Logging, recordings & data

All data is local to the project directory.

**Settings**

- `codex_shell_settings.json` – shell configuration (assistant name, guided mode, etc.).

**Voice profile / calibration**

- `codex_voice_profile.json` – wake-word, command-word aliases, per-word calibration metadata.

**Logs** (under `logs/`)

- `codex_shell_debug.log` – detailed debug info (when debug logging is enabled).
- `codex_shell_session.log` – prompt/response and key events (when logging is enabled).

**Recordings** (under `recordings/`)

- `tts_YYYYMMDD_HHMMSS.wav` – saved TTS audio when “save recordings” is ON.

Toggle these from the shell:

- `v-debug` / `Alice debug on` / `Alice debug off`
- `v-recordings` / `Alice save recordings`
- `v-logprompts` / `Alice log`

---

## License

This project is licensed under the **MIT License** with explicit liability exclusion.

See the full text in [`LICENSE`](LICENSE).

import io
import os
import math
import time
import glob
import base64
import datetime
import torch
import numpy as np
import soundfile as sf
import gradio as gr
from qwen_tts import Qwen3TTSModel

# ---------------------------------------------------------------------------
# Model location
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def find_local_checkpoint() -> str:
    search_patterns = [
        r"D:\hf_cache\hub\models--Qwen--Qwen3-TTS-12Hz-1.7B-VoiceDesign\snapshots\*",
        os.path.join(BASE_DIR, "models_cache", "hub", "models--Qwen--Qwen3-TTS-12Hz-1.7B-VoiceDesign", "snapshots", "*"),
    ]
    for pattern in search_patterns:
        matches = glob.glob(pattern)
        if matches:
            latest = sorted(matches)[-1]
            if os.path.exists(os.path.join(latest, "model.safetensors")):
                return latest
    return "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"


MODEL_PATH = find_local_checkpoint()
print(f"[Studio] Model Checkpoint: {MODEL_PATH}", flush=True)

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.init()
    DEVICE = "cuda:0"
    DTYPE = torch.bfloat16
    GPU_NAME = torch.cuda.get_device_name(0).replace("NVIDIA GeForce ", "")
    VRAM_TOTAL = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
    HW_LABEL = f"{GPU_NAME} · {VRAM_TOTAL:g} GB · bfloat16"
else:
    DEVICE = "cpu"
    DTYPE = torch.float32
    VRAM_TOTAL = 0.0
    HW_LABEL = "CPU · float32"

# Load model once at startup
tts_model = Qwen3TTSModel.from_pretrained(
    MODEL_PATH,
    device_map=DEVICE,
    dtype=DTYPE,
    attn_implementation=None,
)
print("[Studio] Model loaded and ready.", flush=True)

# ---------------------------------------------------------------------------
# Voice Presets
# ---------------------------------------------------------------------------
PRESETS = {
    "Narrator": (
        "A mature male narrator with a deep, warm, resonant voice, measured pacing, calm authoritative delivery, and subtle cinematic presence.",
        "Welcome to your local studio. Every word is synthesized purely on your GPU with pristine fidelity.",
    ),
    "Podcast": (
        "A charismatic female podcast host, energetic pacing, warm friendly tone, articulate, and naturally conversational.",
        "Hey everyone! Welcome back. Today we are exploring local AI voice synthesis running entirely offline.",
    ),
    "Cyber AI": (
        "A calm, soothing female AI assistant with a smooth, precise cadence, subtle futuristic resonance, and clean inflection.",
        "Neural audio synthesis online. All parameters calibrated and ready for generation.",
    ),
    "Storyteller": (
        "An elderly grandfather recounting a nostalgic fairytale, raspy warm timbre, slow gentle pacing, and comforting cadence.",
        "Once upon a time, in a quiet valley beneath the stars, an ancient clock began to tick again.",
    ),
    "News Anchor": (
        "A professional television news anchor, neutral accent, crisp enunciation, formal tempo, and balanced vocal resonance.",
        "Good evening. Local open-source voice models now deliver instant studio quality on consumer hardware.",
    ),
    "Anime Hero": (
        "A spirited young protagonist with a bright, heroic voice, confident inflection, bold presence, and high emotional energy.",
        "No matter what challenges stand in our way, we won't give up. Let's push forward together!",
    ),
}

LANGUAGES = ["English", "Auto", "Chinese", "Japanese", "Korean", "German", "French", "Spanish", "Italian", "Russian"]

# ---------------------------------------------------------------------------
# HTML fragments
# NOTE: gr.HTML does not run <script> tags. All behaviour is attached through
# the JS strings further down (demo.load + event-level js hooks).
# ---------------------------------------------------------------------------
N_BARS = 56


def build_visualizer() -> str:
    bars = []
    for i in range(N_BARS):
        envelope = math.sin(math.pi * (i + 0.5) / N_BARS) ** 0.8
        texture = 0.55 + 0.45 * abs(math.sin(i * 0.9) * math.cos(i * 0.37))
        h = max(8, round(12 + 84 * envelope * texture))
        bars.append(f'<span style="--h:{h}%;--i:{i}"></span>')
    return f"""
    <div id="viz" class="viz">
        <div class="viz-state">
            <span class="s-idle">Ready</span>
            <span class="s-live">Rendering on device</span>
            <span class="s-play">Playing</span>
        </div>
        <div class="viz-bars">{''.join(bars)}</div>
    </div>
    """


def player_empty() -> str:
    return """
    <div class="player empty">
        <span>Your take appears here. Nothing is saved until you download it.</span>
    </div>
    """


def player_html(wav_bytes: bytes, filename: str) -> str:
    # The audio lives only in the page (data URI): no file is written to disk.
    b64 = base64.b64encode(wav_bytes).decode("ascii")
    return f"""
    <div class="player">
        <audio controls autoplay preload="auto" src="data:audio/wav;base64,{b64}"></audio>
        <button type="button" class="dl-btn" data-filename="{filename}">Download WAV</button>
    </div>
    """


def stats_html(duration="—", gen="—", rtf="—", vram="—", note="Waiting for your first take.", kind="idle") -> str:
    return f"""
    <div class="stats {kind}">
        <div class="stat"><b>{duration}</b><i>Audio length</i></div>
        <div class="stat"><b>{gen}</b><i>Render time</i></div>
        <div class="stat"><b>{rtf}</b><i>Real-time factor</i></div>
        <div class="stat"><b>{vram}</b><i>Peak VRAM</i></div>
    </div>
    <div class="stats-note {kind}">{note}</div>
    """


def error_html(msg: str) -> str:
    safe = str(msg).replace("<", "&lt;").replace(">", "&gt;")
    return stats_html(note=safe, kind="error")


# ---------------------------------------------------------------------------
# Synthesis Handler
# ---------------------------------------------------------------------------
def synthesize(text: str, voice_instruct: str, language: str):
    if not text or not text.strip():
        return player_empty(), error_html("Add some text to speak, then try again.")
    if not voice_instruct or not voice_instruct.strip():
        return player_empty(), error_html("Describe the voice you want, then try again.")

    clean_text = text.strip()
    clean_instruct = voice_instruct.strip()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    try:
        wavs, sr = tts_model.generate_voice_design(
            text=clean_text,
            language=language,
            instruct=clean_instruct,
        )
        t_gen = time.time() - t0

        peak_vram = round(torch.cuda.max_memory_allocated() / (1024**3), 2) if torch.cuda.is_available() else 0.0
        raw_audio = wavs[0]

        # float32 numpy array normalised to [-1, 1]
        audio_arr = np.asarray(raw_audio, dtype=np.float32)
        if audio_arr.ndim > 1:
            audio_arr = np.mean(audio_arr, axis=-1)
        max_val = np.max(np.abs(audio_arr)) if audio_arr.size else 0.0
        if max_val > 1.0:
            audio_arr = audio_arr / (max_val + 1e-12)
        audio_arr = np.clip(audio_arr, -1.0, 1.0)

        audio_duration = round(len(audio_arr) / sr, 2)
        rtf = round(t_gen / max(audio_duration, 0.01), 2)

        # Encode a 16-bit PCM WAV in memory only (nothing is saved to disk)
        buf = io.BytesIO()
        sf.write(buf, audio_arr, sr, format="WAV", subtype="PCM_16")
        wav_bytes = buf.getvalue()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"VoiceDesign_{timestamp}.wav"

        print(f"[Done] {audio_duration}s in {t_gen:.2f}s ({rtf}x RTF) • VRAM {peak_vram} GB", flush=True)

        stats = stats_html(
            duration=f"{audio_duration}s",
            gen=f"{t_gen:.1f}s",
            rtf=f"{rtf}×",
            vram=f"{peak_vram} GB" if VRAM_TOTAL else "n/a",
            note="Take ready. Use Download WAV to keep it.",
            kind="ok",
        )
        return player_html(wav_bytes, filename), stats

    except Exception as e:
        err = f"Generation failed: {type(e).__name__} — {e}"
        print(f"[Error] {err}", flush=True)
        return player_empty(), error_html(err)


# ---------------------------------------------------------------------------
# Gradio theme (dark)
# ---------------------------------------------------------------------------
_tokens = dict(
    body_background_fill="#0a0814",
    background_fill_primary="#0a0814",
    background_fill_secondary="rgba(255,255,255,0.03)",
    body_text_color="#ece6ff",
    body_text_color_subdued="#9a90b8",
    block_background_fill="transparent",
    block_border_width="0px",
    block_border_color="transparent",
    block_shadow="none",
    block_label_background_fill="transparent",
    block_title_text_color="#9a90b8",
    panel_background_fill="transparent",
    border_color_primary="rgba(236,230,255,0.10)",
    input_background_fill="rgba(8,6,18,0.65)",
    input_background_fill_focus="rgba(8,6,18,0.85)",
    input_border_color="rgba(236,230,255,0.10)",
    input_border_color_focus="#a78bfa",
    input_border_color_hover="rgba(167,139,250,0.45)",
    input_placeholder_color="#6d6488",
    input_shadow_focus="0 0 0 3px rgba(167,139,250,0.18)",
    button_primary_background_fill="linear-gradient(120deg,#a78bfa 0%,#ff7eb6 55%,#ffb38a 100%)",
    button_primary_background_fill_hover="linear-gradient(120deg,#b9a1ff 0%,#ff93c4 55%,#ffc29f 100%)",
    button_primary_text_color="#16102a",
    button_primary_border_color="transparent",
    button_secondary_background_fill="rgba(255,255,255,0.04)",
    button_secondary_text_color="#d8d0f2",
    button_secondary_border_color="rgba(236,230,255,0.12)",
    checkbox_background_color="rgba(8,6,18,0.65)",
)
_theme_vars = {}
for _k, _v in _tokens.items():
    _theme_vars[_k] = _v
    _theme_vars[_k + "_dark"] = _v

studio_theme = gr.themes.Base(
    primary_hue="violet",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_lg,
    font=["DM Sans", "Inter", "Segoe UI", "system-ui", "sans-serif"],
    font_mono=["JetBrains Mono", "ui-monospace", "Consolas", "monospace"],
).set(**_theme_vars)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
GRAIN = (
    "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'>"
    "<filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='2' stitchTiles='stitch'/>"
    "<feColorMatrix values='0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  0 0 0 .55 0'/></filter>"
    "<rect width='100%' height='100%' filter='url(%23n)'/></svg>\")"
)

aesthetic_css = """
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=DM+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --ink: #0a0814;
    --line: rgba(236, 230, 255, 0.09);
    --text: #ece6ff;
    --mute: #9a90b8;
    --faint: #6d6488;
    --lilac: #a78bfa;
    --rose: #ff7eb6;
    --peach: #ffb38a;
    --well: rgba(6, 4, 14, 0.55);
    --field: rgba(8, 6, 18, 0.65);
    --chip: rgba(255, 255, 255, 0.04);
    --chip-text: #d8d0f2;
    --chip-hover: rgba(167, 139, 250, 0.16);
    --stat-idle: #4f4668;
    --bar-idle: #5b5278;
    --display: 'Bricolage Grotesque', 'DM Sans', system-ui, sans-serif;
}

html, body {
    margin: 0 !important;
    padding: 0 !important;
    height: 100vh !important;
    overflow: hidden !important;
    color: var(--text);
    background:
        radial-gradient(900px 600px at 8% -10%, rgba(167, 139, 250, 0.22), transparent 60%),
        radial-gradient(800px 560px at 100% 110%, rgba(255, 179, 138, 0.16), transparent 60%),
        radial-gradient(600px 400px at 70% 10%, rgba(255, 126, 182, 0.08), transparent 60%),
        var(--ink) !important;
}

/* film grain (never intercepts clicks) */
body::after {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 50;
    opacity: 0.055;
    mix-blend-mode: overlay;
    background-image: __GRAIN__;
}

.gradio-container {
    max-width: 1440px !important;
    height: 100vh !important;
    display: flex !important;
    flex-direction: column !important;
    padding: 18px 28px 20px !important;
    box-sizing: border-box !important;
    overflow: hidden !important;
    background: transparent !important;
    font-family: 'DM Sans', system-ui, sans-serif !important;
    color: var(--text);
}
.gradio-container > .main, .gradio-container .main, .gradio-container .wrap.app { background: transparent !important; }
footer { display: none !important; }

/* ---------- Top bar ---------- */
.top-nav { display: flex; justify-content: space-between; align-items: center; padding: 4px 2px 16px; }
.brand {
    display: flex; align-items: center; gap: 12px;
    font-family: var(--display); font-weight: 700; font-size: 17px; letter-spacing: -0.02em; color: var(--text);
}
.brand-mark {
    width: 30px; height: 30px; border-radius: 10px;
    display: inline-flex; flex-direction: row; align-items: center; justify-content: center;
    background: linear-gradient(135deg, var(--lilac), var(--rose) 60%, var(--peach));
    box-shadow: 0 6px 22px rgba(167, 139, 250, 0.35);
}
.brand-mark i { display: block; width: 3px; border-radius: 3px; background: #16102a; margin: 0 1.5px; }
.brand-mark i:nth-child(1) { height: 8px; }
.brand-mark i:nth-child(2) { height: 15px; }
.brand-mark i:nth-child(3) { height: 11px; }
.brand-model { font-weight: 500; font-size: 12.5px; color: var(--mute); letter-spacing: 0; font-family: 'DM Sans', sans-serif; }

.hw-badge {
    display: inline-flex; align-items: center; gap: 9px;
    font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 12px;
    color: var(--chip-text);
    padding: 7px 14px;
    border: 1px solid var(--line);
    border-radius: 999px;
    background: var(--chip);
    backdrop-filter: blur(10px);
}
.hw-dot { width: 7px; height: 7px; border-radius: 50%; background: #2dd4bf; box-shadow: 0 0 10px #2dd4bf; }

/* ---------- Layout ---------- */
#main-row { flex: 1 1 auto; min-height: 0; gap: 20px !important; align-items: stretch !important; flex-wrap: nowrap !important; }
#left-col { flex: 5 1 0 !important; min-width: 0 !important; }
#right-col { flex: 7 1 0 !important; min-width: 0 !important; }

.studio-card {
    position: relative;
    height: 100% !important;
    box-sizing: border-box !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 10px !important;
    padding: 26px 28px !important;
    border-radius: 22px !important;
    border: 1px solid var(--line) !important;
    background: linear-gradient(165deg, rgba(255, 255, 255, 0.055), rgba(255, 255, 255, 0.012) 55%) !important;
    backdrop-filter: blur(22px);
    box-shadow: 0 30px 80px -30px rgba(0, 0, 0, 0.7), inset 0 1px 0 rgba(255, 255, 255, 0.06) !important;
    overflow: hidden !important;
}
.studio-card .block, .studio-card .form { background: transparent !important; border: none !important; box-shadow: none !important; }

/* ---------- Left: compose ---------- */
.headline {
    font-family: var(--display); font-weight: 800;
    font-size: clamp(28px, 3.1vw, 42px); line-height: 1.02; letter-spacing: -0.035em;
    margin: 2px 0 6px; color: var(--text);
}
.sub { color: var(--mute); font-size: 13.5px; line-height: 1.5; margin: 0 0 6px; max-width: 46ch; }
.field-label { font-size: 12.5px; font-weight: 600; color: var(--mute); margin: 8px 0 -2px; letter-spacing: 0.01em; }

textarea {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    line-height: 1.55 !important;
    color: var(--text) !important;
    background: var(--field) !important;
    border-radius: 14px !important;
    padding: 14px 16px !important;
    transition: border-color .2s ease, box-shadow .2s ease !important;
    scrollbar-width: thin;
    scrollbar-color: var(--bar-idle) transparent;
}
textarea::-webkit-scrollbar { width: 8px; }
textarea::-webkit-scrollbar-track { background: transparent; }
textarea::-webkit-scrollbar-thumb { background: var(--bar-idle); border-radius: 999px; }

/* presets */
.pill-row { display: flex !important; flex-wrap: wrap !important; gap: 7px !important; margin: 0 !important; }
.pill-row > * { flex: 0 0 auto !important; min-width: 0 !important; }
.preset-pill, .preset-pill button {
    min-width: unset !important; width: auto !important; height: 30px !important;
    padding: 0 13px !important;
    border-radius: 999px !important;
    font-size: 12.5px !important; font-weight: 500 !important;
    color: var(--chip-text) !important;
    background: var(--chip) !important;
    border: 1px solid var(--line) !important;
    transition: background .18s ease, border-color .18s ease, color .18s ease !important;
}
.preset-pill:hover, .preset-pill button:hover {
    background: var(--chip-hover) !important;
    border-color: var(--lilac) !important;
    color: var(--text) !important;
}

/* action row */
#action-row { align-items: end !important; gap: 12px !important; margin-top: auto !important; flex-wrap: nowrap !important; }
#lang-col { flex: 1 1 0 !important; min-width: 0 !important; }
#btn-col { flex: 2 1 0 !important; min-width: 0 !important; }
.synthesize-btn {
    height: 50px !important;
    border-radius: 14px !important;
    font-family: var(--display) !important; font-weight: 700 !important; font-size: 15.5px !important; letter-spacing: -0.01em !important;
    border: none !important;
    color: #16102a !important;
    box-shadow: 0 12px 34px -8px rgba(255, 126, 182, 0.55), inset 0 1px 0 rgba(255, 255, 255, 0.35) !important;
    transition: transform .18s ease, box-shadow .18s ease !important;
}
.synthesize-btn:hover { transform: translateY(-2px); box-shadow: 0 18px 40px -8px rgba(255, 126, 182, 0.7), inset 0 1px 0 rgba(255,255,255,.4) !important; }
.synthesize-btn:active { transform: translateY(0) scale(.99); }
.lang-dd input, .lang-dd .wrap-inner { font-size: 14px !important; }

/* ---------- Right: stage ---------- */
.stage-title {
    display: flex; justify-content: space-between; align-items: baseline;
    font-family: var(--display); font-weight: 700; font-size: 20px; letter-spacing: -0.02em; color: var(--text);
}
.stage-title span { font-family: 'DM Sans', sans-serif; font-weight: 500; font-size: 12.5px; color: var(--mute); letter-spacing: 0; }

.viz {
    position: relative;
    height: clamp(180px, 34vh, 340px);
    border-radius: 18px;
    border: 1px solid var(--line);
    background: radial-gradient(60% 90% at 50% 100%, rgba(167, 139, 250, 0.14), transparent 70%), var(--well);
    overflow: hidden;
    display: flex; flex-direction: column;
}
.viz-state {
    position: absolute; top: 14px; left: 18px; z-index: 2;
    font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 11.5px; color: var(--mute);
    display: inline-flex; align-items: center; gap: 8px;
}
.viz-state span::before { content: ""; display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 8px; }
.viz-state .s-idle::before { background: var(--bar-idle); }
.viz-state .s-live::before, .viz-state .s-play::before { background: var(--rose); box-shadow: 0 0 10px var(--rose); }
.viz-state .s-live, .viz-state .s-play { display: none; color: #ffd3e6; }
.viz.live .s-idle, .viz.playing .s-idle { display: none; }
.viz.live .s-live { display: inline; }
.viz.playing:not(.live) .s-play { display: inline; }

.viz-bars { flex: 1; display: flex; align-items: center; justify-content: space-between; gap: 4px; padding: 44px 22px 22px; }
.viz-bars span {
    flex: 1; height: var(--h); min-height: 4px; border-radius: 999px;
    background: linear-gradient(to top, var(--lilac), var(--rose) 60%, var(--peach));
    opacity: 0.42;
    transform-origin: center;
    transition: opacity .4s ease;
}
.viz.live .viz-bars span, .viz.playing .viz-bars span {
    opacity: 1;
    animation: vizpulse 0.9s cubic-bezier(.45, .05, .55, .95) infinite alternate;
    animation-delay: calc(var(--i) * -53ms);
    filter: drop-shadow(0 0 6px rgba(255, 126, 182, 0.45));
}
@keyframes vizpulse {
    from { transform: scaleY(0.18); }
    to   { transform: scaleY(1.08); }
}

/* player */
.player {
    display: flex; align-items: center; gap: 12px;
    padding: 10px 12px;
    border-radius: 16px;
    border: 1px solid var(--line);
    background: var(--well);
}
.player audio { flex: 1; min-width: 0; height: 44px; color-scheme: dark; }
.player.empty { color: var(--faint); font-size: 13px; padding: 16px 18px; min-height: 44px; box-sizing: border-box; }
.dl-btn {
    flex: 0 0 auto;
    height: 40px; padding: 0 18px;
    border-radius: 12px;
    border: 1px solid var(--line);
    background: var(--chip);
    color: var(--chip-text);
    font-family: 'DM Sans', sans-serif; font-weight: 600; font-size: 13px;
    cursor: pointer;
    transition: background .18s ease, border-color .18s ease, color .18s ease;
}
.dl-btn:hover { background: var(--chip-hover); border-color: var(--lilac); color: var(--text); }

/* stats */
.stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
.stat {
    padding: 12px 14px; border-radius: 14px;
    border: 1px solid var(--line);
    background: var(--chip);
    display: flex; flex-direction: column; gap: 4px;
}
.stat b { font-family: var(--display); font-weight: 700; font-size: 22px; letter-spacing: -0.03em; color: var(--text); }
.stat i { font-style: normal; font-size: 11.5px; color: var(--mute); }
.stats.idle .stat b { color: var(--stat-idle); }
.stats-note { margin-top: 10px; font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 11.5px; color: var(--mute); }
.stats-note.ok { color: #5eead4; }
.stats-note.error { color: #ff9db5; font-family: 'DM Sans', sans-serif; font-size: 13px; }

.fine-print { margin-top: auto; display: flex; justify-content: space-between; gap: 12px; font-size: 11.5px; color: var(--faint); }

/* ---------- Motion + small screens ---------- */
@media (prefers-reduced-motion: reduce) {
    .viz.live .viz-bars span, .viz.playing .viz-bars span { animation: none; transform: none; }
    .synthesize-btn, .preset-pill { transition: none !important; }
}
@media (max-width: 960px) {
    html, body, .gradio-container { height: auto !important; min-height: 100vh; overflow: auto !important; }
    #main-row { flex-direction: column !important; }
    .studio-card { height: auto !important; }
    .stats { grid-template-columns: repeat(2, 1fr); }
    .hw-badge { display: none; }
}
:focus-visible { outline: 2px solid var(--lilac) !important; outline-offset: 2px; }
""".replace("__GRAIN__", GRAIN)

# ---------------------------------------------------------------------------
# JavaScript hooks
# ---------------------------------------------------------------------------
# Runs once on page load. Uses document-level listeners so it keeps working
# when Gradio replaces the player HTML after every take.
JS_INIT = """
() => {
    if (window.__studioInit) return;
    window.__studioInit = true;

    const viz = () => document.getElementById('viz');

    // Visualizer follows real playback (media events don't bubble, so capture)
    document.addEventListener('play', (e) => {
        if (e.target && e.target.tagName === 'AUDIO') { const v = viz(); if (v) v.classList.add('playing'); }
    }, true);
    ['pause', 'ended'].forEach((evt) => {
        document.addEventListener(evt, (e) => {
            if (e.target && e.target.tagName === 'AUDIO') { const v = viz(); if (v) v.classList.remove('playing'); }
        }, true);
    });

    // Explicit download: turn the in-page audio into a file only on click
    document.addEventListener('click', async (e) => {
        const btn = e.target && e.target.closest ? e.target.closest('.dl-btn') : null;
        if (!btn) return;
        const audio = document.querySelector('.player audio');
        if (!audio || !audio.src) return;
        try {
            const blob = await (await fetch(audio.src)).blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = btn.dataset.filename || 'VoiceDesign.wav';
            document.body.appendChild(a);
            a.click();
            a.remove();
            setTimeout(() => URL.revokeObjectURL(url), 4000);
        } catch (err) {
            console.error('Download failed', err);
        }
    });
}
"""

# Runs in the browser right before synthesize(); must return the inputs.
JS_START = """
(text, voice, lang) => {
    const v = document.getElementById('viz');
    if (v) { v.classList.remove('playing'); v.classList.add('live'); }
    return [text, voice, lang];
}
"""

# Runs after synthesize() finishes (success or error).
JS_STOP = """
() => {
    const v = document.getElementById('viz');
    if (v) v.classList.remove('live');
}
"""


def make_loader(name: str):
    def _load():
        return PRESETS[name]
    return _load


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Voice Design Studio") as demo:
    gr.HTML(f"""
    <div class="top-nav">
        <div class="brand">
            <span class="brand-mark"><i></i><i></i><i></i></span>
            <span>Voice Design Studio</span>
            <span class="brand-model">Qwen3-TTS 1.7B</span>
        </div>
        <div class="hw-badge"><span class="hw-dot"></span>{HW_LABEL} · offline</div>
    </div>
    """)

    with gr.Row(elem_id="main-row", equal_height=True):
        # ------------------- Left: compose -------------------
        with gr.Column(elem_id="left-col", elem_classes=["studio-card"]):
            gr.HTML("""
                <div class="headline">Describe a voice.<br>Hear it in seconds.</div>
                <p class="sub">Write who is speaking, how they sound, and what they say. Everything renders on your machine.</p>
            """)

            gr.HTML('<div class="field-label">Voice description</div>')
            voice_desc = gr.Textbox(
                show_label=False,
                lines=3,
                max_lines=3,
                placeholder="Age, timbre, pacing, accent, emotion…",
                value=PRESETS["Narrator"][0],
            )

            with gr.Row(elem_classes=["pill-row"]):
                preset_buttons = {name: gr.Button(name, elem_classes=["preset-pill"], size="sm") for name in PRESETS}

            gr.HTML('<div class="field-label">Script</div>')
            text_input = gr.Textbox(
                show_label=False,
                lines=4,
                max_lines=4,
                placeholder="Type what the voice should say…",
                value=PRESETS["Narrator"][1],
            )

            with gr.Row(elem_id="action-row"):
                with gr.Column(elem_id="lang-col", min_width=120):
                    lang_select = gr.Dropdown(
                        show_label=False,
                        choices=LANGUAGES,
                        value="English",
                        container=False,
                        elem_classes=["lang-dd"],
                    )
                with gr.Column(elem_id="btn-col", min_width=160):
                    gen_btn = gr.Button("Generate voice", variant="primary", elem_classes=["synthesize-btn"])

        # ------------------- Right: stage -------------------
        with gr.Column(elem_id="right-col", elem_classes=["studio-card"]):
            gr.HTML('<div class="stage-title">Stage<span>Latest take plays automatically</span></div>')
            gr.HTML(build_visualizer())

            player_out = gr.HTML(player_empty())
            status_html = gr.HTML(stats_html())

            gr.HTML("""
            <div class="fine-print">
                <span>24 kHz · 16-bit PCM WAV</span>
                <span>Takes are saved only when you download them</span>
            </div>
            """)

    # One-time page setup (playback-linked visualizer + explicit download)
    demo.load(fn=None, js=JS_INIT)

    # Preset clicks
    for _name, _btn in preset_buttons.items():
        _btn.click(make_loader(_name), outputs=[voice_desc, text_input])

    # Generate
    gen_btn.click(
        fn=synthesize,
        js=JS_START,
        inputs=[text_input, voice_desc, lang_select],
        outputs=[player_out, status_html],
        show_progress_on=[player_out],
    ).then(fn=None, js=JS_STOP)

if __name__ == "__main__":
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=studio_theme,
        css=aesthetic_css,
    )
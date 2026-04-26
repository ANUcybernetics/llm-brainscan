"""Train a character-level GPT with live speech input and weight visualisation.

On startup, loads Shakespeare as the initial training corpus. A microphone
listener runs in the background; when speech is detected, it is transcribed
via Whisper, run through the model as an inference pass (shown in the text
strip), and then added to the training data. The model's brain is displayed
on an 8K canvas, one pixel per parameter.
"""

import argparse
import datetime as dt
import hashlib
import json
import logging
import threading
import time
from pathlib import Path

import numpy as np
import torch

from brainscan.captions import CaptionsState, compose_caption
from brainscan.conversation import (
    Conversation,
    ConversationState,
    ListenerSnapshot,
)
from brainscan.data import TextBuffer, prepare_batches
from brainscan.layout import (
    HEIGHT,
    LAYOUT_HEIGHT,
    TEXT_STRIP_HEIGHT,
    WIDTH,
    compute_layout,
    default_sections,
    layout_summary,
    layout_to_flat_order,
)
from brainscan.model import GPT
from brainscan.rebirth import RebirthScheduler, rebirth, rotate_audience_log
from brainscan.renderer import (
    CaptionsFrame,
    LaneFrame,
    LiveRenderer,
    OffscreenRenderer,
    flatten_weights,
)
from brainscan.snapshot import capture_weights
from brainscan.tts import TTSEngine

log = logging.getLogger(__name__)

AUDIENCE_HEIGHT = 90
MODEL_LANE_HEIGHT = 90
CAPTIONS_HEIGHT = 12


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def download_shakespeare(data_dir: Path) -> Path:
    path = data_dir / "shakespeare.txt"
    if path.exists():
        return path
    import urllib.request

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, path)
    return path


def save_frame(canvas: np.ndarray, path: Path) -> None:
    from PIL import Image

    img = Image.fromarray(canvas[:, :, :3])
    img.save(path)


def _build_weight_buffer(
    weights: dict[str, torch.Tensor],
    flat_order: list[str],
    canvas_pixels: int,
) -> np.ndarray:
    np_weights = {k: v.cpu().numpy() for k, v in weights.items()}
    flat, count = flatten_weights(np_weights, layout_order=flat_order)
    buf = np.zeros(canvas_pixels, dtype=np.float32)
    n = min(count, canvas_pixels)
    buf[:n] = flat[:n]
    return buf


def _build_lane_frames(
    convo: Conversation,
    captions_state: CaptionsState,
    commit_pulse: float = 0.0,
    partial_pulse: float = 0.0,
) -> tuple[LaneFrame, LaneFrame, CaptionsFrame]:
    a_chars, a_attrs, _ = convo.audience.snapshot()
    m_chars, _, m_probs = convo.model_lane.snapshot()

    model_caret = (
        convo.model_lane.count
        if convo.state in (ConversationState.MUSE, ConversationState.RESPONDING)
        else -1
    )

    audience = LaneFrame(
        chars=a_chars,
        attrs_or_probs=a_attrs,
        count=convo.audience.count,
        pulse=commit_pulse,
        edge_pulse=partial_pulse,
    )
    model = LaneFrame(
        chars=m_chars,
        attrs_or_probs=m_probs,
        count=convo.model_lane.count,
        caret_col=model_caret,
    )
    cap_chars = compose_caption(captions_state)
    captions = CaptionsFrame(chars=cap_chars, count=len(cap_chars))
    return audience, model, captions


def _current_event_line(
    now: float, event_holder: dict[str, object]
) -> str:
    """Return the event line if it has not yet expired, else clear and return ''."""
    expires = float(event_holder["expires_at"])  # type: ignore[arg-type]
    if now >= expires:
        event_holder["text"] = ""
        return ""
    return str(event_holder["text"])


def _process_committed(
    listener: ListenerSnapshot, training: TextBuffer
) -> None:
    for text in listener.committed:
        log.info("Audience: %s", text)
        training.append(text)


def _decay_pulse(holder: dict, now_t: float, half_life: float = 0.5) -> float:
    """Decay a pulse holder by elapsed wall-clock time and return the new value."""
    dt = now_t - holder["last_render_t"]
    if dt > 0.0:
        holder["value"] = max(0.0, holder["value"] - dt / half_life)
    holder["last_render_t"] = now_t
    return holder["value"]


def _caption_state_label(convo: Conversation, step: int, loss: float) -> str:
    if convo.state == ConversationState.LISTENING:
        return "listening..."
    if convo.state == ConversationState.RESPONDING:
        return "responding..."
    return f"musing | step {step:,} loss {loss:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Brainscan training")
    parser.add_argument(
        "--data", type=str, default=None, help="Path to training text file"
    )
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=9)
    parser.add_argument("--n-embd", type=int, default=558)
    parser.add_argument("--sequence-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=0, help="0 = run forever")
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=10,
        help="Capture/render every N steps",
    )
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save weight visualisation frames as PNGs",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Display weights live in a fullscreen window",
    )
    parser.add_argument(
        "--no-mic",
        action="store_true",
        help="Disable microphone input",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="small",
        help="Whisper model size for STT",
    )
    parser.add_argument(
        "--whisper-device",
        type=str,
        default="cpu",
        help="Device for Whisper inference (cpu or cuda)",
    )
    parser.add_argument(
        "--gen-tokens",
        type=int,
        default=200,
        help="Tokens to generate per inference pass",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=0.01,
        help="RMS threshold for speech detection (raise for noisy environments)",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=2.0,
        help="Audio chunk size in seconds (debounce window)",
    )
    parser.add_argument(
        "--min-speech-seconds",
        type=float,
        default=0.5,
        help="Minimum speech duration to trigger transcription",
    )
    parser.add_argument(
        "--max-speech-seconds",
        type=float,
        default=30.0,
        help="Maximum speech duration before forced transcription",
    )
    parser.add_argument(
        "--speak", action="store_true", help="Enable TTS for model responses"
    )
    parser.add_argument(
        "--drone",
        action="store_true",
        help="Enable sub-bass drone tracking training loss",
    )
    parser.add_argument(
        "--rebirth-at",
        type=str,
        default=None,
        help="Daily rebirth time HH:MM (24h, local). Default off.",
    )
    parser.add_argument(
        "--persist-audience-days",
        type=int,
        default=7,
        help="Number of past audience-log days to prepend on rebirth (0 to disable)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed for daily rebirth (per-day seed = hash(date)+base)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    if args.save_images:
        frames_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    data_path = Path(args.data) if args.data else download_shakespeare(data_dir)
    raw_data = data_path.read_bytes()

    audience_log = output_dir / "audience_input.txt"
    training_data = TextBuffer(raw_data, persist_path=audience_log)
    print(f"Dataset: {len(training_data):,} bytes from {data_path}")

    model = GPT(
        vocab_size=256,
        sequence_len=args.sequence_len,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    ).to(device)

    param_counts = {name: p.numel() for name, p in model.named_parameters()}
    total_params = sum(param_counts.values())
    print(f"Model: {total_params:,} parameters")
    print(
        f"Display: {WIDTH}x{HEIGHT} = {WIDTH * HEIGHT:,} pixels"
        f" (layout: {WIDTH}x{LAYOUT_HEIGHT}, text strip: {TEXT_STRIP_HEIGHT}px)"
    )
    print(f"Utilisation: {total_params / (WIDTH * HEIGHT) * 100:.1f}%")

    sections = default_sections(n_layer=args.n_layer)
    layout = compute_layout(param_counts, sections=sections)
    print(layout_summary(layout))

    layout_path = output_dir / "layout.json"
    layout_path.write_text(
        json.dumps(
            {name: rect.to_dict() for name, rect in layout.items()}, indent=2
        )
    )
    print(f"\nLayout saved to {layout_path}")

    flat_order = layout_to_flat_order(layout)

    offscreen_renderer = None
    if args.save_images:
        offscreen_renderer = OffscreenRenderer(
            WIDTH,
            HEIGHT,
            audience_height=AUDIENCE_HEIGHT,
            model_height=MODEL_LANE_HEIGHT,
            captions_height=CAPTIONS_HEIGHT,
        )
        print(f"Offscreen renderer initialised ({WIDTH}x{HEIGHT})")

    live_renderer = None
    if args.live:
        live_renderer = LiveRenderer(
            WIDTH,
            HEIGHT,
            audience_height=AUDIENCE_HEIGHT,
            model_height=MODEL_LANE_HEIGHT,
            captions_height=CAPTIONS_HEIGHT,
            fullscreen=True,
        )
        print(f"Live renderer initialised ({WIDTH}x{HEIGHT})")

    listener = None
    if not args.no_mic:
        from brainscan.stt import SpeechConfig, SpeechListener

        stt_config = SpeechConfig(
            model_size=args.whisper_model,
            device=args.whisper_device,
            chunk_seconds=args.chunk_seconds,
            silence_threshold=args.silence_threshold,
            min_speech_seconds=args.min_speech_seconds,
            max_speech_seconds=args.max_speech_seconds,
        )
        listener = SpeechListener(config=stt_config)
        listener.start()
        print("Microphone listener started")

    drone = None
    if args.drone:
        from brainscan.audio_drone import DroneOscillator

        drone = DroneOscillator()
        drone.start()

    def train_loop() -> None:
        nonlocal training_data
        optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)

        convo = Conversation(
            tts_enabled=args.speak,
            response_token_count=args.gen_tokens,
        )

        partial_holder: dict[str, str] = {"text": ""}
        partial_pulse_holder: dict[str, float] = {"value": 0.0, "last_render_t": 0.0}
        commit_pulse_holder: dict[str, float] = {"value": 0.0, "last_render_t": 0.0}
        event_holder: dict[str, object] = {"text": "", "expires_at": 0.0}

        def _show_event(text: str, duration: float = 5.0, now: float = 0.0) -> None:
            event_holder["text"] = text
            event_holder["expires_at"] = now + duration

        def on_partial(text: str) -> None:
            partial_holder["text"] = text
            partial_pulse_holder["value"] = 1.0
            partial_pulse_holder["last_render_t"] = time.time() - t0

        def on_speech_end() -> None:
            partial_holder["text"] = ""

        if listener is not None:
            listener._partial_callback = on_partial
            listener._speech_end_callback = on_speech_end

        gen_iter = model.streaming_generate(
            b"ROMEO: ", device=device, emit_prompt=False
        )

        def token_fn(_now: float):
            return next(gen_iter)

        rebirth_sched = RebirthScheduler(
            at_hh_mm=args.rebirth_at,
            state_path=output_dir / "rebirth.last",
        )
        tts = TTSEngine(enabled=args.speak)

        print("\nTraining started (Ctrl+C to stop)...")
        t0 = time.time()
        step = 0
        prev_state = ConversationState.MUSE
        rebirth_state: dict[str, object] = {"phase": "idle", "started_at": 0.0}
        global_brightness = 1.0

        try:
            while True:
                if args.steps > 0 and step >= args.steps:
                    break

                now_t = time.time() - t0

                committed: list[str] = []
                if listener is not None:
                    committed = listener.get_text()

                # Reset gen_iter BEFORE convo.step so the first RESPONDING
                # token is sampled from the response-seeded generator
                # rather than the muse one.
                if committed:
                    partial_holder["text"] = ""
                    gen_iter.close()
                    seed = committed[-1].encode("utf-8", errors="replace")
                    gen_iter = model.streaming_generate(
                        seed, device=device, emit_prompt=False
                    )

                snapshot = ListenerSnapshot(
                    committed=committed,
                    partial=partial_holder["text"] or None,
                    in_speech=bool(partial_holder["text"])
                    or bool(committed),
                )
                events = convo.step(
                    now=now_t,
                    listener=snapshot,
                    token_fn=token_fn,
                )
                for ev in events.speak_events:
                    duration = tts.speak(ev.text)
                    if duration > 0.0:
                        # extend cooldown so listener doesn't re-trigger on TTS playback
                        convo._cooldown_until = max(convo._cooldown_until, now_t + duration)
                _process_committed(snapshot, training_data)

                # detect LISTENING → RESPONDING transition (commit)
                if (
                    prev_state == ConversationState.LISTENING
                    and convo.state == ConversationState.RESPONDING
                ):
                    commit_pulse_holder["value"] = 1.0
                prev_state = convo.state

                # one optimiser step per loop turn
                x, y = prepare_batches(
                    training_data, args.batch_size, args.sequence_len, device
                )
                _, loss = model(x, y)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                # rebirth state machine
                rebirth_phase = str(rebirth_state["phase"])
                if rebirth_phase == "idle":
                    now_dt = dt.datetime.now()
                    if rebirth_sched.due(now_dt):
                        rebirth_state = {"phase": "fading_out", "started_at": now_t}
                        global_brightness = 1.0
                elif rebirth_phase == "fading_out":
                    elapsed = now_t - float(rebirth_state["started_at"])
                    global_brightness = max(0.0, 1.0 - elapsed / 2.0)
                    if elapsed >= 2.0:
                        now_dt = dt.datetime.now()
                        yesterday = now_dt.date() - dt.timedelta(days=1)
                        rotate_audience_log(
                            audience_log, output_dir / "audience", yesterday
                        )
                        res = rebirth(
                            model=model,
                            seed_corpus=raw_data,
                            audience_dir=output_dir / "audience",
                            persist_days=args.persist_audience_days,
                            seed=int.from_bytes(
                                hashlib.sha256(
                                    f"{now_dt.date().isoformat()}-{args.seed}".encode()
                                ).digest()[:4],
                                "big",
                            ),
                        )
                        training_data = TextBuffer(
                            res.corpus, persist_path=audience_log
                        )
                        optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)
                        rebirth_sched.mark_fired(now_dt)
                        (output_dir / "rebirth.log").open("a").write(
                            f"{now_dt.date().isoformat()} seed={res.seed}"
                            f" persist_days={args.persist_audience_days}\n"
                        )
                        _show_event(f"dawn {now_dt.strftime('%H:%M')}", now=now_t)
                        rebirth_state = {"phase": "fading_in", "started_at": now_t}
                elif rebirth_phase == "fading_in":
                    elapsed = now_t - float(rebirth_state["started_at"])
                    global_brightness = min(1.0, elapsed / 2.0)
                    if elapsed >= 2.0:
                        global_brightness = 1.0
                        rebirth_state = {"phase": "idle", "started_at": 0.0}
                else:
                    global_brightness = 1.0

                if step % args.snapshot_every == 0:
                    current_weights = capture_weights(model)
                    if drone is not None:
                        drone.update_loss(loss.item())
                    dt_elapsed = time.time() - t0
                    print(
                        f"step {step:5d} | loss {loss.item():.4f}"
                        f" | data {len(training_data):,}B"
                        f" | {dt_elapsed:.1f}s | {convo.state.value}"
                    )

                    captions_state = CaptionsState(
                        state_label=_caption_state_label(convo, step, loss.item()),
                        cursor_label="",
                        event_line=_current_event_line(now_t, event_holder),
                    )
                    commit_val = _decay_pulse(commit_pulse_holder, now_t)
                    partial_val = _decay_pulse(partial_pulse_holder, now_t)
                    audience, model_frame, captions = _build_lane_frames(
                        convo,
                        captions_state,
                        commit_pulse=commit_val,
                        partial_pulse=partial_val,
                    )

                    if offscreen_renderer is not None:
                        canvas_pixels = (
                            offscreen_renderer.width
                            * offscreen_renderer.height
                        )
                        buf = _build_weight_buffer(
                            current_weights, flat_order, canvas_pixels
                        )
                        canvas = offscreen_renderer.render(
                            buf,
                            audience=audience,
                            model=model_frame,
                            captions=captions,
                            global_brightness=global_brightness,
                        )
                        save_frame(
                            canvas, frames_dir / f"frame_{step:06d}.png"
                        )

                    if live_renderer is not None:
                        canvas_pixels = WIDTH * HEIGHT
                        buf = _build_weight_buffer(
                            current_weights, flat_order, canvas_pixels
                        )
                        live_renderer.update(
                            buf,
                            audience=audience,
                            model=model_frame,
                            captions=captions,
                            global_brightness=global_brightness,
                        )

                step += 1

        except KeyboardInterrupt:
            print("\nStopping...")

        if listener is not None:
            listener.stop()

        if drone is not None:
            drone.stop()

        if live_renderer is not None:
            live_renderer.close()

        total_time = time.time() - t0
        if step > 0:
            print(
                f"Done. {step} steps in {total_time:.1f}s"
                f" ({step / total_time:.1f} steps/s)"
            )

    if live_renderer is not None:
        train_thread = threading.Thread(target=train_loop, daemon=True)
        train_thread.start()
        live_renderer.run()
    else:
        train_loop()


if __name__ == "__main__":
    main()

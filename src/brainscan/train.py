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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from brainscan import tuning
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
    compute_text_overlays,
    default_sections,
    layout_summary,
    place_weights_on_canvas,
)
from brainscan.model import GPT
from brainscan.rebirth import (
    RebirthFade,
    RebirthPhase,
    RebirthScheduler,
    rebirth,
    rotate_audience_log,
    step_rebirth_phase,
)
from brainscan.renderer import (
    LaneFrame,
    LiveRenderer,
    OffscreenRenderer,
)
from brainscan.snapshot import capture_weights
from brainscan.tts import TTSEngine

log = logging.getLogger(__name__)

_train_state_history: list[ConversationState] | None = None
"""Optional list for tests to capture the convo state after each step."""


@dataclass
class PulseState:
    """Thread-safe pulse value with time-based decay."""
    _value: float = 0.0
    _last_render_t: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def value(self) -> float:
        with self._lock:
            return self._value

    def trigger(self, now_t: float) -> None:
        with self._lock:
            self._value = 1.0
            self._last_render_t = now_t

    def decay(self, now_t: float, half_life: float = tuning.PULSE_HALF_LIFE_SECONDS) -> float:
        with self._lock:
            dt_elapsed = now_t - self._last_render_t
            if dt_elapsed > 0.0:
                self._value = max(0.0, self._value - dt_elapsed / half_life)
            self._last_render_t = now_t
            return self._value


AUDIENCE_HEIGHT = 112
MODEL_LANE_HEIGHT = 112


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
    layout: dict,
    width: int,
    height: int,
) -> np.ndarray:
    """Place weights at their layout rects on a `(height, width)` canvas, then
    flatten to the row-major float32 array the renderer expects."""
    np_weights = {k: v.cpu().numpy() for k, v in weights.items()}
    canvas = place_weights_on_canvas(np_weights, layout, width=width, height=height)
    return canvas.ravel()


def _build_lane_frames(
    convo: Conversation,
    commit_pulse: float = 0.0,
    partial_pulse: float = 0.0,
) -> tuple[LaneFrame, LaneFrame]:
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
    return audience, model


def _process_committed(
    listener: ListenerSnapshot, training: TextBuffer
) -> None:
    for text in listener.committed:
        log.info("Audience: %s", text)
        training.append(text)


def _build_listener(args) -> object | None:
    """Construct a SpeechListener (unstarted) so the caller can wire
    callbacks before audio capture begins. Returns None when --no-mic."""
    if args.no_mic:
        return None
    from brainscan.stt import SpeechConfig, SpeechListener

    stt_config = SpeechConfig(
        model_size=args.whisper_model,
        device=args.whisper_device,
        chunk_seconds=args.chunk_seconds,
        silence_threshold=args.silence_threshold,
        min_speech_seconds=args.min_speech_seconds,
        max_speech_seconds=args.max_speech_seconds,
        audio_device=args.audio_device,
    )
    return SpeechListener(config=stt_config)


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Brainscan training")
    parser.add_argument(
        "--data", type=str, default=None, help="Path to training text file"
    )
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=9)
    parser.add_argument("--n-embd", type=int, default=540)
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
        default=tuning.SILENCE_THRESHOLD,
        help="RMS threshold for speech detection (raise for noisy environments)",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=tuning.STT_CHUNK_SECONDS,
        help="Audio capture chunk in seconds (sets VAD/end-of-speech latency)",
    )
    parser.add_argument(
        "--min-speech-seconds",
        type=float,
        default=tuning.MIN_SPEECH_SECONDS,
        help="Minimum speech duration to trigger transcription",
    )
    parser.add_argument(
        "--max-speech-seconds",
        type=float,
        default=tuning.MAX_SPEECH_SECONDS,
        help="Maximum speech duration before forced transcription",
    )
    parser.add_argument(
        "--audio-device",
        type=int,
        default=None,
        help="sounddevice input device index (None = system default)",
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
        help="Weekly rebirth time 'DOW HH:MM' (e.g., 'MON 02:00'). Default off.",
    )
    parser.add_argument(
        "--persist-audience-cycles",
        type=int,
        default=tuning.PERSIST_AUDIENCE_CYCLES_DEFAULT,
        help="Number of past audience-log cycles (~weeks) to prepend on rebirth (0 to disable)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed for weekly rebirth (per-week seed = hash(iso-week+base))",
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
    seed_corpus = np.memmap(data_path, dtype=np.uint8, mode="r")

    audience_log = output_dir / "audience_input.txt"
    training_data = TextBuffer(b"", persist_path=audience_log)
    print(
        f"Dataset: {len(seed_corpus):,} seed bytes from {data_path}"
        f" + {len(training_data):,} audience bytes"
    )

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
    layout = compute_layout(
        param_counts,
        sections=sections,
        section_gutter=tuning.LAYOUT_SECTION_GUTTER_PX,
        group_gutter=tuning.LAYOUT_GROUP_GUTTER_PX,
        item_gutter=tuning.LAYOUT_ITEM_GUTTER_PX,
        label_gap_px=tuning.LAYOUT_LABEL_GAP_PX,
        section_label_height=tuning.LAYOUT_SECTION_LABEL_PX,
        min_section_width=tuning.LAYOUT_MIN_SECTION_WIDTH,
    )
    print(layout_summary(layout))

    layout_path = output_dir / "layout.json"
    layout_path.write_text(
        json.dumps(
            {name: rect.to_dict() for name, rect in layout.items()}, indent=2
        )
    )
    print(f"\nLayout saved to {layout_path}")

    offscreen_renderer = None
    if args.save_images:
        offscreen_renderer = OffscreenRenderer(
            WIDTH,
            HEIGHT,
            audience_height=AUDIENCE_HEIGHT,
            model_height=MODEL_LANE_HEIGHT,
        )
        print(f"Offscreen renderer initialised ({WIDTH}x{HEIGHT})")

    live_renderer = None
    if args.live:
        live_renderer = LiveRenderer(
            WIDTH,
            HEIGHT,
            audience_height=AUDIENCE_HEIGHT,
            model_height=MODEL_LANE_HEIGHT,
            fullscreen=True,
        )
        print(f"Live renderer initialised ({WIDTH}x{HEIGHT})")

    overlays = compute_text_overlays(layout, sections)
    if offscreen_renderer is not None:
        offscreen_renderer.set_overlays(overlays)
    if live_renderer is not None:
        live_renderer.set_overlays(overlays)

    listener = _build_listener(args)

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
        partial_pulse = PulseState()
        commit_pulse = PulseState()

        gen_iter = model.streaming_generate(
            b"ROMEO: ", device=device, emit_prompt=False
        )

        def token_fn(_now: float):
            return next(gen_iter)

        rebirth_sched = RebirthScheduler(
            at_dow_hh_mm=args.rebirth_at,
            state_path=output_dir / "rebirth.last",
        )
        tts = TTSEngine(enabled=args.speak)

        print("\nTraining started (Ctrl+C to stop)...")
        t0 = time.time()
        step = 0
        prev_state = ConversationState.MUSE
        rebirth_fade = RebirthFade()
        global_brightness = 1.0

        def on_partial(text: str) -> None:
            partial_holder["text"] = text
            partial_pulse.trigger(time.time() - t0)

        def on_speech_end() -> None:
            partial_holder["text"] = ""

        if listener is not None:
            listener.partial_callback = on_partial
            listener.speech_end_callback = on_speech_end
            listener.start()
            print("Microphone listener started")

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
                if _train_state_history is not None:
                    _train_state_history.append(convo.state)
                # tts.speak() returns immediately; sd.play() under the hood replaces any
                # in-flight stream with the new one. Back-to-back responses can't actually
                # overlap because cooldown_seconds + duration prevents the listener from
                # re-triggering until playback ends.
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
                    commit_pulse.trigger(now_t)
                prev_state = convo.state

                # one optimiser step per loop turn
                x, y = prepare_batches(
                    training_data,
                    args.batch_size,
                    args.sequence_len,
                    device,
                    seed_corpus=seed_corpus,
                    seed_weight=tuning.SEED_CORPUS_SAMPLING_WEIGHT,
                )
                _, loss = model(x, y)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                # rebirth state machine
                is_due = (
                    rebirth_fade.phase == RebirthPhase.IDLE
                    and rebirth_sched.due(dt.datetime.now())
                )
                rebirth_fade, global_brightness, should_perform = step_rebirth_phase(
                    rebirth_fade, now_t, is_due,
                    fade_duration=tuning.REBIRTH_FADE_DURATION_SECONDS,
                )
                if should_perform:
                    now_dt = dt.datetime.now()
                    iso = now_dt.date().isocalendar()
                    iso_key = f"{iso.year}-W{iso.week:02d}"
                    # Rotate accumulated audience log under the Monday of the
                    # week that just closed (the model that just died).
                    prev_week_monday = dt.date.fromisocalendar(
                        iso.year, iso.week, 1
                    ) - dt.timedelta(days=7)
                    rotate_audience_log(
                        audience_log, output_dir / "audience", prev_week_monday
                    )
                    res = rebirth(
                        model=model,
                        audience_dir=output_dir / "audience",
                        persist_count=args.persist_audience_cycles,
                        seed=int.from_bytes(
                            hashlib.sha256(
                                f"{iso_key}-{args.seed}".encode()
                            ).digest()[:4],
                            "big",
                        ),
                    )
                    training_data = TextBuffer(res.audience, persist_path=audience_log)
                    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)
                    rebirth_sched.mark_fired(now_dt)
                    (output_dir / "rebirth.log").open("a").write(
                        f"{iso_key} seed={res.seed}"
                        f" persist_cycles={args.persist_audience_cycles}\n"
                    )

                    # Recompute layout/overlays in case n_embd changed.
                    new_param_counts = {n: p.numel() for n, p in model.named_parameters()}
                    if new_param_counts != param_counts:
                        new_layout = compute_layout(
                            new_param_counts,
                            sections=sections,
                            section_gutter=tuning.LAYOUT_SECTION_GUTTER_PX,
                            group_gutter=tuning.LAYOUT_GROUP_GUTTER_PX,
                            item_gutter=tuning.LAYOUT_ITEM_GUTTER_PX,
                            label_gap_px=tuning.LAYOUT_LABEL_GAP_PX,
                            section_label_height=tuning.LAYOUT_SECTION_LABEL_PX,
                            min_section_width=tuning.LAYOUT_MIN_SECTION_WIDTH,
                        )
                        layout.clear()
                        layout.update(new_layout)
                        new_overlays = compute_text_overlays(layout, sections)
                        if offscreen_renderer is not None:
                            offscreen_renderer.set_overlays(new_overlays)
                        if live_renderer is not None:
                            live_renderer.set_overlays(new_overlays)

                if step % args.snapshot_every == 0:
                    current_weights = capture_weights(model)
                    if drone is not None:
                        drone.update_loss(loss.item())
                    dt_elapsed = time.time() - t0
                    print(
                        f"step {step:5d} | loss {loss.item():.4f}"
                        f" | seed {len(seed_corpus):,}B"
                        f" | aud {len(training_data):,}B"
                        f" | {dt_elapsed:.1f}s | {convo.state.value}"
                    )

                    commit_val = commit_pulse.decay(now_t)
                    partial_val = partial_pulse.decay(now_t)
                    audience, model_frame = _build_lane_frames(
                        convo,
                        commit_pulse=commit_val,
                        partial_pulse=partial_val,
                    )

                    if offscreen_renderer is not None:
                        buf = _build_weight_buffer(
                            current_weights,
                            layout,
                            offscreen_renderer.width,
                            offscreen_renderer.height,
                        )
                        canvas = offscreen_renderer.render(
                            buf,
                            audience=audience,
                            model=model_frame,
                            global_brightness=global_brightness,
                        )
                        save_frame(
                            canvas, frames_dir / f"frame_{step:06d}.png"
                        )

                    if live_renderer is not None:
                        buf = _build_weight_buffer(
                            current_weights, layout, WIDTH, HEIGHT,
                        )
                        live_renderer.update(
                            buf,
                            audience=audience,
                            model=model_frame,
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

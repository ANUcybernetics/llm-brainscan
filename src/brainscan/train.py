"""Train a character-level GPT with live speech input and weight visualisation.

On startup, loads Shakespeare as the initial training corpus. A microphone
listener runs in the background; when speech is detected, it is transcribed
via Whisper, run through the model as an inference pass (shown in the text
strip), and then added to the training data. The model's brain is displayed
on an 8K canvas, one pixel per parameter.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from brainscan.data import TextBuffer, decode, prepare_batches
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
from brainscan.renderer import (
    LiveRenderer,
    OffscreenRenderer,
    flatten_weights,
)
from brainscan.snapshot import capture_weights

log = logging.getLogger(__name__)


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


def prepare_display_buffers(
    weights: dict[str, torch.Tensor],
    flat_order: list[str],
    canvas_pixels: int,
    text_chars: list[int] | None = None,
    text_probs: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    np_weights = {k: v.cpu().numpy() for k, v in weights.items()}
    flat, count = flatten_weights(np_weights, layout_order=flat_order)
    buf = np.zeros(canvas_pixels, dtype=np.float32)
    n = min(count, canvas_pixels)
    buf[:n] = flat[:n]
    chars = np.array(text_chars, dtype=np.uint32) if text_chars else None
    probs = np.array(text_probs, dtype=np.float32) if text_probs else None
    return buf, chars, probs


def render_frame(
    renderer: OffscreenRenderer,
    weights: dict[str, torch.Tensor],
    flat_order: list[str],
    text_chars: list[int] | None = None,
    text_probs: list[float] | None = None,
) -> np.ndarray:
    canvas_pixels = renderer.width * renderer.height
    buf, chars, probs = prepare_display_buffers(
        weights, flat_order, canvas_pixels, text_chars, text_probs
    )
    return renderer.render(buf, text_chars=chars, text_probs=probs)


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
            WIDTH, HEIGHT, text_strip_height=TEXT_STRIP_HEIGHT
        )
        print(f"Offscreen renderer initialised ({WIDTH}x{HEIGHT})")

    live_renderer = None
    if args.live:
        live_renderer = LiveRenderer(
            WIDTH,
            HEIGHT,
            text_strip_height=TEXT_STRIP_HEIGHT,
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

    def train_loop() -> None:
        optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)

        gen_chars: list[int] = []
        gen_probs: list[float] = []

        print("\nTraining started (Ctrl+C to stop)...")
        t0 = time.time()
        step = 0

        try:
            while True:
                if args.steps > 0 and step >= args.steps:
                    break

                if listener is not None:
                    new_texts = listener.get_text()
                    for text in new_texts:
                        log.info("Audience: %s", text)
                        prompt = text.encode("utf-8", errors="replace")
                        gen_chars, gen_probs = model.generate(
                            prompt, args.gen_tokens, device
                        )
                        generated = decode(gen_chars)
                        log.info("Response: %s", generated[:200])
                        training_data.append(text)

                x, y = prepare_batches(
                    training_data, args.batch_size, args.sequence_len, device
                )
                _, loss = model(x, y)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                if step % args.snapshot_every == 0:
                    current_weights = capture_weights(model)

                    dt = time.time() - t0
                    data_size = len(training_data)
                    print(
                        f"step {step:5d} | loss {loss.item():.4f}"
                        f" | data {data_size:,}B | {dt:.1f}s"
                    )

                    if not gen_chars:
                        gen_chars, gen_probs = model.generate(
                            b"ROMEO: ",
                            args.gen_tokens,
                            device,
                        )

                    if offscreen_renderer is not None:
                        canvas = render_frame(
                            offscreen_renderer,
                            current_weights,
                            flat_order,
                            gen_chars,
                            gen_probs,
                        )
                        save_frame(canvas, frames_dir / f"frame_{step:06d}.png")

                    if live_renderer is not None:
                        buf, chars, probs = prepare_display_buffers(
                            current_weights,
                            flat_order,
                            WIDTH * HEIGHT,
                            gen_chars,
                            gen_probs,
                        )
                        live_renderer.update(buf, text_chars=chars, text_probs=probs)

                step += 1

        except KeyboardInterrupt:
            print("\nStopping...")

        if listener is not None:
            listener.stop()
            print("Microphone listener stopped")

        if live_renderer is not None:
            live_renderer.close()

        total_time = time.time() - t0
        if step > 0:
            print(
                f"Done. {step} steps in {total_time:.1f}s"
                f" ({step / total_time:.1f} steps/s)"
            )

    if live_renderer is not None:
        import threading

        train_thread = threading.Thread(target=train_loop, daemon=True)
        train_thread.start()
        live_renderer.run()
    else:
        train_loop()


if __name__ == "__main__":
    main()

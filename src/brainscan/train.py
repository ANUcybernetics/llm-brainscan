"""Train a character-level GPT and capture weight snapshots each step."""

import argparse
import json
import time
from pathlib import Path

import torch
import numpy as np

from brainscan.model import GPT
from brainscan.data import prepare_batches, load_text_dataset, decode
from brainscan.snapshot import capture_weights, capture_weight_deltas, ActivationCapture
from brainscan.layout import compute_layout, weights_to_image, WIDTH, HEIGHT


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
    img = Image.fromarray(canvas)
    img.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Brainscan training")
    parser.add_argument("--data", type=str, default=None, help="Path to training text file")
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=9)
    parser.add_argument("--n-embd", type=int, default=576)
    parser.add_argument("--sequence-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--snapshot-every", type=int, default=10, help="Save weight snapshot every N steps")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--save-images", action="store_true", help="Save weight visualisation frames as PNGs")
    args = parser.parse_args()

    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    if args.save_images:
        frames_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    if args.data:
        data_path = Path(args.data)
    else:
        data_path = download_shakespeare(data_dir)
    data = load_text_dataset(data_path)
    print(f"Dataset: {len(data):,} bytes from {data_path}")

    model = GPT(
        vocab_size=256,
        sequence_len=args.sequence_len,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    ).to(device)

    param_groups = model.param_count()
    print(f"Model: {param_groups['total']:,} parameters")
    print(f"Display: {WIDTH}x{HEIGHT} = {WIDTH * HEIGHT:,} pixels")
    print(f"Utilisation: {param_groups['total'] / (WIDTH * HEIGHT) * 100:.1f}%")
    for name, count in param_groups.items():
        if name != "total":
            print(f"  {name}: {count:,}")

    layout = compute_layout(param_groups)
    layout_path = output_dir / "layout.json"
    layout_path.write_text(json.dumps(layout, indent=2))
    print(f"Layout saved to {layout_path}")

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)
    prev_weights = capture_weights(model)

    print(f"\nTraining for {args.steps} steps...")
    t0 = time.time()

    for step in range(args.steps):
        x, y = prepare_batches(data, args.batch_size, args.sequence_len, device)
        _, loss = model(x, y)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if step % args.snapshot_every == 0 or step == args.steps - 1:
            current_weights = capture_weights(model)
            deltas = capture_weight_deltas(current_weights, prev_weights)
            prev_weights = current_weights

            dt = time.time() - t0
            print(f"step {step:5d} | loss {loss.item():.4f} | {dt:.1f}s")

            if args.save_images:
                canvas = weights_to_image(current_weights, layout)
                save_frame(canvas, frames_dir / f"weights_{step:06d}.png")

                delta_canvas = weights_to_image(deltas, layout)
                save_frame(delta_canvas, frames_dir / f"deltas_{step:06d}.png")

    total_time = time.time() - t0
    print(f"\nDone. {args.steps} steps in {total_time:.1f}s ({args.steps / total_time:.1f} steps/s)")

    print("\nRunning inference with activation capture...")
    cap = ActivationCapture(model)
    cap.install()
    prompt = b"ROMEO: "
    tokens = list(prompt)
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _ = model(x)
    print(f"Captured {len(cap.activations)} activation tensors")
    for name, act in cap.activations.items():
        print(f"  {name}: {list(act.shape)}")
    cap.remove()

    print(f"\nSample generation from prompt '{prompt.decode()}':")
    model.eval()
    context = torch.tensor([tokens], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(200):
            logits, _ = model(context[:, -args.sequence_len :])
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat([context, next_token], dim=1)
    generated = decode(context[0].tolist())
    print(generated)


if __name__ == "__main__":
    main()

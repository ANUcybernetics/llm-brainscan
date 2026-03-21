import torch
from conftest import SMALL_CONFIG

from brainscan.model import GPT
from brainscan.snapshot import ActivationCapture, capture_weight_deltas, capture_weights


class TestCaptureWeights:
    def test_captures_all_parameters(self, small_model):
        weights = capture_weights(small_model)
        param_names = {name for name, _ in small_model.named_parameters()}
        assert set(weights.keys()) == param_names

    def test_weights_are_detached_clones(self, small_model):
        weights = capture_weights(small_model)
        for name, tensor in weights.items():
            assert not tensor.requires_grad
            # Verify it's a clone by mutating and checking original is unchanged
            original = dict(small_model.named_parameters())[name].detach().clone()
            tensor.zero_()
            current = dict(small_model.named_parameters())[name]
            assert torch.equal(current.detach(), original)


class TestCaptureWeightDeltas:
    def test_delta_is_zero_for_same_weights(self, small_model):
        w = capture_weights(small_model)
        deltas = capture_weight_deltas(w, w)
        for name, delta in deltas.items():
            assert torch.all(delta == 0), f"Non-zero delta for {name}"

    def test_delta_after_training_step(self, small_model, device):
        before = capture_weights(small_model)
        x = torch.randint(0, 256, (2, 16), device=device)
        y = torch.randint(0, 256, (2, 16), device=device)
        _, loss = small_model(x, y)
        loss.backward()
        optimiser = torch.optim.AdamW(small_model.parameters(), lr=1e-3)
        optimiser.step()
        after = capture_weights(small_model)
        deltas = capture_weight_deltas(after, before)
        any_changed = any(not torch.all(d == 0) for d in deltas.values())
        assert any_changed, "No weights changed after training step"


class TestActivationCapture:
    def test_captures_activations(self, small_model, device):
        cap = ActivationCapture(small_model)
        cap.install()
        x = torch.randint(0, 256, (1, 8), device=device)
        with torch.no_grad():
            small_model(x)
        assert len(cap.activations) > 0
        cap.remove()

    def test_activation_shapes(self, small_model, device):
        cap = ActivationCapture(small_model)
        cap.install()
        x = torch.randint(0, 256, (1, 8), device=device)
        with torch.no_grad():
            small_model(x)
        for name, act in cap.activations.items():
            assert act.ndim >= 2, f"Activation {name} has unexpected shape {act.shape}"
        cap.remove()

    def test_activations_are_detached(self, small_model, device):
        cap = ActivationCapture(small_model)
        cap.install()
        x = torch.randint(0, 256, (1, 8), device=device)
        with torch.no_grad():
            small_model(x)
        for name, act in cap.activations.items():
            assert not act.requires_grad, f"Activation {name} still requires grad"
        cap.remove()

    def test_clear(self, small_model, device):
        cap = ActivationCapture(small_model)
        cap.install()
        x = torch.randint(0, 256, (1, 8), device=device)
        with torch.no_grad():
            small_model(x)
        assert len(cap.activations) > 0
        cap.clear()
        assert len(cap.activations) == 0
        cap.remove()

    def test_remove_hooks(self, small_model, device):
        cap = ActivationCapture(small_model)
        cap.install()
        cap.remove()
        cap.clear()
        x = torch.randint(0, 256, (1, 8), device=device)
        with torch.no_grad():
            small_model(x)
        assert len(cap.activations) == 0


class TestTrainingLoop:
    def test_loss_decreases(self, device):
        model = GPT(**SMALL_CONFIG).to(device)
        optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
        # Use a very simple repeating pattern that should be easy to learn
        pattern = b"abcabc" * 200
        from brainscan.data import prepare_batches

        losses = []
        for _ in range(50):
            x, y = prepare_batches(
                pattern,
                batch_size=8,
                sequence_len=SMALL_CONFIG["sequence_len"],
                device=device,
            )
            _, loss = model(x, y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_snapshot_during_training(self, device):
        model = GPT(**SMALL_CONFIG).to(device)
        optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
        data = b"hello world " * 200
        from brainscan.data import prepare_batches

        prev = capture_weights(model)
        snapshots = [prev]
        for _step in range(5):
            x, y = prepare_batches(
                data,
                batch_size=4,
                sequence_len=SMALL_CONFIG["sequence_len"],
                device=device,
            )
            _, loss = model(x, y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            current = capture_weights(model)
            capture_weight_deltas(current, prev)
            snapshots.append(current)
            prev = current
        assert len(snapshots) == 6
        # Verify weights actually changed across steps
        first_param = next(iter(snapshots[0].keys()))
        assert not torch.equal(snapshots[0][first_param], snapshots[-1][first_param])

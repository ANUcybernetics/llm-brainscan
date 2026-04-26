from brainscan.audio_drone import DroneOscillator


def test_constructs_with_defaults():
    drone = DroneOscillator()
    assert drone.sample_rate == 44100
    assert drone.min_hz == 40.0
    assert drone.max_hz == 60.0
    assert drone._stream is None


def test_update_loss_changes_hz():
    drone = DroneOscillator(min_hz=40.0, max_hz=60.0)
    drone.update_loss(loss=0.0, max_loss=4.0)
    hz_low_loss = drone._hz
    drone.update_loss(loss=4.0, max_loss=4.0)
    hz_high_loss = drone._hz
    # at zero loss (max smartness) -> high pitch (60 Hz); at max loss -> low pitch (40 Hz)
    assert hz_low_loss == 60.0
    assert hz_high_loss == 40.0


def test_update_loss_clips_above_max():
    drone = DroneOscillator()
    drone.update_loss(loss=100.0, max_loss=4.0)
    assert drone._hz == drone.min_hz

#!/bin/sh
# Pick the first USB input source as the PulseAudio default and crank
# its volume to 100%. Run as ExecStartPre before brainscan starts so
# the STT listener binds to the right microphone (the Jetson's on-board
# 3.5mm input has a noise floor far above the silence threshold and
# isn't usable as a mic).
#
# Falls back gracefully (exit 0) if no USB mic is present so a tethered
# operator can still start the service without one.

set -eu

# graphical-session.target should bring pulseaudio up before us, but
# add a short retry in case we win the race.
for _ in 1 2 3 4 5; do
    pactl info >/dev/null 2>&1 && break
    sleep 1
done

source_name=$(pactl list sources short \
    | awk '$2 ~ /^alsa_input/ && tolower($2) ~ /(rode|usb)/ {print $2; exit}')

if [ -z "$source_name" ]; then
    echo "select-audio-source: no USB input found; leaving default unchanged" >&2
    exit 0
fi

pactl set-default-source "$source_name"
pactl set-source-volume "$source_name" 100%
echo "select-audio-source: default = $source_name @ 100%"

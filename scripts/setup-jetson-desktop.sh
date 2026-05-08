#!/usr/bin/env bash
# Suppress Ubuntu/Jetson desktop popups that overlay the brainscan fullscreen
# window during exhibition. Run once per fresh Jetson, as the exhibition user
# (jane) --- it touches the user's autostart and apt.
#
# Usage:  ./setup-jetson-desktop.sh
#
# What it does:
#   1. Installs xdotool (used to dismiss any sticky notification banners that
#      slip through, e.g. "Caution: Hot surface. Do Not Touch.").
#   2. Adds Hidden=true overrides under ~/.config/autostart/ for the three
#      autostart entries that have caused on-screen popups in production:
#        - update-notifier        ("Software updater" prompt)
#        - gnome-software-service ("Software" background service / popups)
#        - nvpmodel_indicator     (NVIDIA's "Caution: Hot surface" banner)
#      User-level overrides take precedence over /etc/xdg/autostart/ and
#      survive package upgrades.

set -euo pipefail

if [[ "${EUID}" -eq 0 ]]; then
  echo "Run as the exhibition user (e.g. jane), not root --- this writes to ~/.config/autostart/." >&2
  exit 1
fi

echo "[1/2] Installing xdotool..."
if command -v xdotool >/dev/null 2>&1; then
  echo "  xdotool already installed: $(xdotool --version | head -1)"
else
  sudo apt-get update
  sudo apt-get install -y xdotool
fi

echo "[2/2] Hiding desktop autostart popups..."
mkdir -p "${HOME}/.config/autostart"
for entry in update-notifier.desktop gnome-software-service.desktop nvpmodel_indicator.desktop; do
  src="/etc/xdg/autostart/${entry}"
  dst="${HOME}/.config/autostart/${entry}"
  if [[ ! -f "${src}" ]]; then
    echo "  skip ${entry} (not present in /etc/xdg/autostart)"
    continue
  fi
  cp "${src}" "${dst}"
  if ! grep -q '^Hidden=true$' "${dst}"; then
    printf '\nHidden=true\n' >> "${dst}"
  fi
  echo "  hid ${entry}"
done

echo
echo "Done. Log out and back in (or reboot) for the autostart changes to take full effect."
echo "Currently-running popup processes can be killed with:"
echo "  pkill -f 'update-manager|update-notifier|nvpmodel_indicator'"

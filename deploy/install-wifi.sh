#!/bin/sh
# Register a WPA2-Enterprise (PEAP/MSCHAPv2) WiFi profile with NetworkManager
# so the Jetson autoconnects on boot when no ethernet is plugged in. Run once
# per site, by hand, after editing the placeholders below. The resulting
# profile lives at /etc/NetworkManager/system-connections/<SSID>.nmconnection
# (root-only, 0600) and persists across reboots and reflashes-with-backup.
#
# Wired remains the preferred route when plugged (lower default metric);
# this profile takes over the moment ethernet drops, and Tailscale roams
# with it. See CLAUDE.md "Deployment on Jetson" for the full chain.
#
# Usage:
#   1. Edit SSID, IDENTITY, PASSWORD below for the target site.
#   2. Uncomment the nmcli block.
#   3. Run on the Jetson: sudo ./deploy/install-wifi.sh
#
# Credentials are NOT checked into the repo. For permanent installs prefer
# loading them from a local untracked env block (mise on Linux) and feeding
# them in via the environment rather than editing this file in place.

set -eu

# --- edit me ------------------------------------------------------------
# SSID='ANU-Secure'
# IDENTITY='your-account'
# PASSWORD='your-password'
# IFACE='wlP1p1s0'          # confirm with: nmcli device status
# -----------------------------------------------------------------------

# sudo nmcli connection add type wifi con-name "$SSID" ifname "$IFACE" ssid "$SSID" -- \
#     wifi-sec.key-mgmt wpa-eap \
#     802-1x.eap peap \
#     802-1x.phase2-auth mschapv2 \
#     802-1x.identity "$IDENTITY" \
#     802-1x.password "$PASSWORD" \
#     connection.autoconnect yes \
#     connection.autoconnect-priority 10
#
# sudo nmcli connection up "$SSID"
# nmcli -g IP4.ADDRESS device show "$IFACE"

echo "install-wifi: this is a template. Edit SSID/IDENTITY/PASSWORD and uncomment the nmcli block before running." >&2
exit 1

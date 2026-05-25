---
id: TASK-3
title: Run the exhibition display at native 8K
status: To Do
assignee: []
created_date: '2026-05-21 23:49'
updated_date: '2026-05-25 06:22'
labels:
  - deployment
  - jetson
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Jetson currently drives the Samsung QN900D 85-inch 8K panel (model QA85QN900DWXXY) at a stable 4K@60, but the artwork's one-pixel-per-parameter design wants native 8K (7680x4320).

The panel connects via a Cable Matters Unidirectional 8K DisplayPort-to-HDMI 2.1 active cable (Amazon AU B094XR43M5) on HDMI 4, with Input Signal Plus enabled for that port.

Attempts so far (both have ruled out the Jetson + X stack as the bottleneck; the cable-to-TV HDMI conversion is the problem):

- 2026-05-22, 8K@30: the Jetson's DP link trained at 8K and X drove 7680x4320 fine, but the active cable did not relay a valid 8K HDMI signal to the TV (panel reported no signal).
- 2026-05-25, 8K@60 with DSC: ran `xrandr --output DP-0 --mode 7680x4320 --rate 59.94`. xrandr exited 0, the X root window resized to 7680x4320, DP-0 showed `7680x4320 59.94*` active, and brainscan came up cleanly ("Live renderer initialised (7680x4320)") and rendered the full canvas at ~2 steps/s (a framebuffer screenshot confirmed the whole 8K canvas was painted correctly). The panel still went dark on the modeset --- same no-signal failure as 8K@30. DSC at 60 Hz did not help the cable handshake. Reverted to 3840x2160@60.

Remaining leads, in order:

1. Run the Cable Matters Windows firmware updater on the cable --- earlier research flagged this as the known fix for 8K handshake failures on the Samsung 8K line. Requires physical access to the cable plus a Windows machine.
2. Once 8K works live, persist the modeset: add an ExecStartPre xrandr call plus an XAUTHORITY environment line to deploy/brainscan.service, so native 8K survives reboots and X restarts (otherwise X reverts to the panel's preferred 4K@60).

Non-urgent: 4K@60 is a fine interim state. At native 8K, brainscan training throughput drops to ~2 steps/s (vs ~3.8 at 4K) --- confirmed on 2026-05-25, expected GPU fragment-fill cost, acceptable for the piece.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Jetson drives the Samsung panel at 7680x4320 and the TV displays a valid picture (no signal-loss or blank screen)
- [ ] #2 brainscan renders fullscreen at native 7680x4320
- [ ] #3 Native 8K survives a clean reboot with no manual intervention, returning to 7680x4320 automatically
- [ ] #4 README deployment section documents the Samsung QN900D TV setup: the Cable Matters cable, HDMI port, Input Signal Plus, and how to set and persist native 8K
<!-- AC:END -->

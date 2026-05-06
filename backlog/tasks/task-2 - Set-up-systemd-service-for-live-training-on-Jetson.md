---
id: TASK-2
title: Set up systemd service for live training on Jetson
status: To Do
assignee: []
created_date: '2026-05-06 07:10'
updated_date: '2026-05-06 07:11'
labels:
  - deployment
  - jetson
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The exhibition piece runs `python -m brainscan.train --live` continuously on the Jetson Orin (jane@jetson-orin, /home/jane/projects/llm-brainscan). Currently it is started by hand on display :0 and survives crashes only as long as no one closes the terminal. For exhibition reliability we want a service that auto-starts on boot, restarts on crash, attaches to the user's X session (DISPLAY=:0), and writes logs to a known location.

Use a user-level systemd unit under /home/jane/.config/systemd/user/ with `loginctl enable-linger jane` so it runs without an active session, OR a system-level unit that uses `su - jane -c` and inherits the X authority --- pick whichever is simpler given the existing autologin configuration on the Jetson.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A systemd unit file is checked into the repo under deploy/ (or similar) so it can be reproduced
- [ ] #2 The service starts python -m brainscan.train --live with DISPLAY=:0 and the right working directory
- [ ] #3 The service auto-starts on boot (linger enabled if user-level)
- [ ] #4 The service auto-restarts on crash with a sensible backoff
- [ ] #5 Logs go to a stable path (e.g. /var/log/brainscan/ or via journalctl) and are reviewable with a single command
- [ ] #6 README has a short '\''Deployment on Jetson'\'' section with the install + enable + status + logs commands
<!-- AC:END -->

---
id: TASK-1
title: Design speech I/O UX for text strip and training loop
status: To Do
assignee: []
created_date: '2026-03-21 03:03'
labels:
  - design
  - stt
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design how speech-to-text input integrates with the display and training loop. The SpeechListener and TextBuffer infrastructure is already implemented; this task covers the UX and control flow decisions that remain before wiring STT into train.py.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Decide how spoken text appears in the text strip (visual distinction between user input and model-generated output, e.g. colour, prefix, separate row)
- [ ] #2 Design mode switching: how the system chooses between using transcribed speech as training data augmentation vs inference prompt
- [ ] #3 Decide whether STT is always-on or opt-in (e.g. --enable-stt flag, or runtime toggle)
- [ ] #4 Design training loop integration: whether speech events interrupt for immediate inference or queue text for training data, and how these interact with snapshot timing
- [ ] #5 Document the design decisions in a short design doc or README section
<!-- AC:END -->

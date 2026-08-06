# SO101 Teleop — Status & Handoff

**Last updated:** 2026-07-15 · **Machine:** `anvil-workcell` (Ubuntu 24.04.4, x86_64) · **Status:** ✅ **WORKING** — leader→follower teleop verified at a steady 60 Hz with zero USB drops.

---

## TL;DR to resume
1. Both arms must be plugged **directly into the PC** (NOT a USB hub — see gotcha below).
2. Launch teleop (single-token command, avoids terminal paste bug):
   ```
   sg dialout -c /home/anvil/SO101_policy/teleop.sh
   ```
   After you've logged out/in once since setup, `dialout` is permanent and you can drop `sg`:
   ```
   bash ~/SO101_policy/teleop.sh
   ```
   (`teleop.sh` self-activates the uv-managed `.venv` — no manual env activation needed.)
3. On start the follower enables torque and snaps to the leader's pose — **pre-match the arms, hands clear.** Ctrl-C to stop.

---

## What got set up (all done, persists across reboot)
- **uv-managed `.venv`** (Python 3.12, system `/usr/bin/python3.12`) at `~/SO101_policy/.venv`. Built with `uv venv --python 3.12` + `uv pip install --torch-backend=cpu -e "./lerobot[feetech]"`. LeRobot cloned at `~/SO101_policy/lerobot`. Torch is **CPU-only** (no CUDA) — fine for teleop. _(Migrated off conda/miniforge 2026-07-15; conda fully removed.)_
- **Serial-retry patch** applied to `.../motors/feetech/feetech.py` (`disable_torque`/`_disable_torque`/`enable_torque` now default `num_retry=3`). Verified.
- **Calibration** copied to `~/.cache/huggingface/lerobot/calibration/{robots/so_follower,teleoperators/so_leader}/`.
- **`anvil` added to `dialout`** group (serial access). Effective in fresh logins; use `sg dialout -c '<cmd>'` in an old session.
- **Stable udev names by USB serial** (`/etc/udev/rules.d/99-so101.rules`):
  | Role | CH343 serial | Stable name |
  |------|--------------|-------------|
  | Leader (teleop) | `5B14111261` | `/dev/so101_leader` |
  | Follower (robot) | `5B14111358` | `/dev/so101_follower` |
  Both adapters are `1a86:55d3`. `teleop.sh` uses these names and self-activates the `.venv`.
  ⚠️ The original Mac `teleop.sh` had leader/follower **swapped** (assumed follower=ACM0); the Linux version is corrected.

## Helper scripts in `~/SO101_policy/`
- `teleop.sh` — the teleop launcher (self-activates env; uses stable `/dev/so101_*` names).
- `probe_follower_read.py [port] [secs]` — read-only 60 Hz probe, no torque. Great for isolating USB stability. Run via `sg dialout -c 'source ~/SO101_policy/.venv/bin/activate && python ~/SO101_policy/probe_follower_read.py /dev/so101_follower'`.
- `monitor.sh` / `monitor_follower.py` — continuous auto-reconnecting monitor that timestamps every USB drop.
- `99-so101.rules` — the udev rules (already installed to `/etc/udev/rules.d/`).

---

## 🔴 The big lesson: USB hub was the culprit
The follower repeatedly connected, ran ~1 teleop loop, then died with
`ConnectionError: ... There is no status packet!` / `SerialException: [Errno 19] No such device`.

**Root cause:** both arms were on a cheap **bus-powered USB hub** (`1a40:0101` "USB 2.0 Hub", also carrying 2 gs_usb CAN adapters). Kernel log showed `usb 1-7-portN: disabled by hub (EMI?), re-enabling...` on repeat — the hub was disabling the follower's port. It worked on the Mac because it was plugged in **direct**.

**Fix:** moved both arms to **direct PC USB ports** → instantly stable (600/600 clean probe reads, teleop steady 60 Hz).

**Ruled out and NOT the cause** (don't re-chase these): dialout perms, conda env, ModemManager, port identity, motor current/torque, servos/daisy-chain, the USB cable (swap test moved nothing), USB autosuspend (`power/control=on` already).

**If USB re-enumeration ever returns:** run `lsusb -t` first. If an arm hangs off a hub, move it direct to the PC before touching anything else. Confirm with `journalctl -k | grep -i "disabled by hub"`.

## Environment quirks
- **Terminal paste bug:** this terminal leaks bracketed-paste markers (`^[[200~`), splitting multi-line/long pastes. Prefer single short path tokens or let Claude run commands. (ModemManager was `stop`ped during debugging; it auto-starts on reboot — harmless now that arms are direct, but can be masked if desired.)

## Next steps / ideas (not started)
- Log out/in (or reboot) once so `dialout` is permanent and `sg` is no longer needed.
- Optionally add cameras + record datasets, then train/run a policy (would want a GPU — none detected on this box).

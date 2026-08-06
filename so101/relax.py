#!/usr/bin/env python3
"""Make the SO101 follower limp so you can move it by hand.

Torque goes OFF immediately -> the arm is free to reposition by hand. Then:
  * press Enter  -> LOCK it holding the new pose (torque back on), or
  * press Ctrl-C -> leave it limp.
"""
from __future__ import annotations

import os

from lerobot.robots.so_follower import SO101Follower
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig

PORT = os.environ.get("FOLLOWER_PORT", "/dev/so101_follower")
robot = SO101Follower(SO101FollowerConfig(port=PORT, id="follower"))
robot.connect()
robot.bus.disable_torque()
print("\nTORQUE OFF — the arm is free. Lift it off the table to a safe, upright pose.")
try:
    input("Press Enter to LOCK it holding there, or Ctrl-C to leave it limp... ")
    robot.bus.enable_torque()
    robot.config.disable_torque_on_disconnect = False
    print("TORQUE ON — holding this pose.")
except KeyboardInterrupt:
    robot.config.disable_torque_on_disconnect = True
    print("\nLeaving the arm limp.")
finally:
    robot.disconnect()

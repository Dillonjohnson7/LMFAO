"""Live follower USB monitor: read-only at 60 Hz, auto-reconnects after each drop,
timestamps every drop so you can correlate with physical actions (reseat DC power,
wiggle the USB connector at the follower board). Ctrl-C to stop.

Usage: python monitor_follower.py [port] [seconds]"""
import sys, time
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/so101_follower"
DURATION = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0

motors = {n: Motor(i, "sts3215", MotorNormMode.DEGREES) for i, n in enumerate(
    ["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll"], start=1)}
motors["gripper"] = Motor(6, "sts3215", MotorNormMode.RANGE_0_100)

period = 1.0 / 60.0
t0 = time.perf_counter()
drops = []
print(f"Monitoring {PORT} at 60 Hz for {DURATION:.0f}s — reseat DC power / wiggle the "
      f"follower board's USB connector and watch for DROP lines. Ctrl-C to stop.")
try:
    while time.perf_counter() - t0 < DURATION:
        try:
            bus = FeetechMotorsBus(port=PORT, motors=motors)
            bus.connect()
        except Exception as e:
            print(f"[{time.perf_counter()-t0:6.1f}s] cannot open port ({type(e).__name__}); retry in 1s")
            time.sleep(1.0)
            continue
        alive_start = time.perf_counter()
        last_beat = 0
        while time.perf_counter() - t0 < DURATION:
            t = time.perf_counter()
            try:
                bus.sync_read("Present_Position", normalize=False, num_retry=0)
            except Exception as e:
                dead = time.perf_counter() - t0
                alive = time.perf_counter() - alive_start
                drops.append(dead)
                print(f"[{dead:6.1f}s] *** DROP *** after {alive:.2f}s alive — {type(e).__name__}")
                try: bus.disconnect()
                except Exception: pass
                time.sleep(0.5)  # let it re-enumerate
                break
            beat = int(time.perf_counter() - alive_start)
            if beat != last_beat:
                last_beat = beat
                print(f"[{time.perf_counter()-t0:6.1f}s] alive {beat:3d}s (continuous)")
            dt = time.perf_counter() - t
            if dt < period:
                time.sleep(period - dt)
except KeyboardInterrupt:
    print("\nstopped by user")
finally:
    el = time.perf_counter() - t0
    print(f"\nSUMMARY: {len(drops)} drop(s) in {el:.0f}s. Drop times(s): "
          f"{', '.join(f'{d:.1f}' for d in drops) if drops else '(none — stable!)'}")

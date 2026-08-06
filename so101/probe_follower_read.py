"""Read-only follower probe: poll Present_Position at 60 Hz, no torque, no writes.
Survives  -> USB drop is caused by motor current draw (power delivery).
Dies here -> the USB link/adapter itself can't hold up (independent of motors)."""
import sys, time
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/so101_follower"

motors = {
    "shoulder_pan":  Motor(1, "sts3215", MotorNormMode.DEGREES),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
    "elbow_flex":    Motor(3, "sts3215", MotorNormMode.DEGREES),
    "wrist_flex":    Motor(4, "sts3215", MotorNormMode.DEGREES),
    "wrist_roll":    Motor(5, "sts3215", MotorNormMode.DEGREES),
    "gripper":       Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}
bus = FeetechMotorsBus(port=PORT, motors=motors)
bus.connect()
print(f"connected read-only to {PORT} (torque NOT enabled)")

period = 1.0 / 60.0
n_ok = n_fail = 0
t0 = time.perf_counter()
try:
    for i in range(600):  # ~10 s at 60 Hz
        t = time.perf_counter()
        try:
            bus.sync_read("Present_Position", normalize=False, num_retry=0)
            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"[iter {i}] READ FAILED after {time.perf_counter()-t0:.2f}s: {type(e).__name__}: {e}")
            break
        if i % 60 == 0:
            print(f"[{i:3d}] ok={n_ok} elapsed={time.perf_counter()-t0:4.1f}s")
        dt = time.perf_counter() - t
        if dt < period:
            time.sleep(period - dt)
finally:
    el = time.perf_counter() - t0
    print(f"RESULT: ok={n_ok} fail={n_fail} elapsed={el:.1f}s avg_rate={n_ok/max(el,1e-9):.1f}Hz")
    try:
        bus.disconnect()
        print("clean disconnect")
    except Exception as e:
        print("disconnect error:", type(e).__name__, e)

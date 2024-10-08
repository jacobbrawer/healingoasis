import asyncio
import time
import numpy as np
import math
from bleak import BleakClient, BleakScanner
from scipy import signal
from collections import deque

# Polar H10 UUIDs
PMD_SERVICE = "fb005c80-02e7-f387-1cad-8acd2d8df0c8"
PMD_CONTROL = "fb005c81-02e7-f387-1cad-8acd2d8df0c8"
PMD_DATA = "fb005c82-02e7-f387-1cad-8acd2d8df0c8"
DEVICE_NAME = "Polar H10"

# LED configuration
LED_COUNT = 150

class PolarH10:
    def __init__(self, device):
        self.device = device
        self.client = None
        self.active_leds = 0  # Number of active LEDs

        # Accelerometer parameters
        self.ACC_SAMPLE_RATE = 200  # Hz
        self.ACC_UPDATE_LOOP_PERIOD = 0.01  # s
        self.GRAVITY_ALPHA = 0.999  # Exponential mean filter for gravity
        self.ACC_MEAN_ALPHA = 0.98  # Exponential mean filter for noise
        self.acc_principle_axis = np.array([0, 0, 1])  # Positive z-axis is out of sensor unit

        # Breathing signal parameters
        self.BR_ACC_SAMPLE_RATE = 10  # Hz
        self.BR_ACC_HIST_SIZE = 1200  # 2 minutes of data at 10 Hz
        self.BR_MAX_FILTER = 30  # breaths per minute maximum
        self.MOVEMENT_THRESHOLD = 0.1  # Threshold for detecting large movements

        # Calibration parameters
        self.CALIBRATION_DURATION = 30  # seconds
        self.BREATH_CYCLE_DURATION = 8  # seconds (4 in, 4 out)
        self.is_calibrating = True
        self.calibration_start_time = None
        self.calibration_samples = []
        self.breath_min = None
        self.breath_max = None
        self.last_printed_led_count = -1  # To avoid printing the same number repeatedly

        # Initialization
        self.acc_gravity = np.full(3, np.nan)
        self.acc_zero_centred_exp_mean = np.zeros(3)
        self.t_last_breath_acc_update = 0
        self.breath_acc_hist = np.full(self.BR_ACC_HIST_SIZE, np.nan)
        self.breath_acc_times = np.full(self.BR_ACC_HIST_SIZE, np.nan)
        self.br_last_phase = 0
        self.current_br = 0

        # Bandpass filter for breathing signal
        self.lowcut = 0.1  # Hz
        self.highcut = 0.5  # Hz
        self.filter_order = 3
        self.nyquist_freq = self.BR_ACC_SAMPLE_RATE / 2
        self.low = self.lowcut / self.nyquist_freq
        self.high = self.highcut / self.nyquist_freq
        self.b, self.a = signal.butter(self.filter_order, [self.low, self.high], btype='band')

    async def connect(self):
        self.client = BleakClient(self.device)
        await self.client.connect()
        print(f"Connected: {self.client.is_connected}")

    async def disconnect(self):
        await self.client.disconnect()

    async def start_acc_stream(self):
        await self.client.write_gatt_char(PMD_CONTROL, bytearray([0x02, 0x02, 0x00, 0x01, 0xC8, 0x00, 0x01, 0x01, 0x10, 0x00, 0x02, 0x01, 0x08, 0x00]), response=True)
        await self.client.start_notify(PMD_DATA, self.acc_data_handler)
        print("ACC stream started")
        print(f"Calibration started. Follow the breath guide for {self.CALIBRATION_DURATION} seconds.")
        print("Breathe in for 4 seconds as the number increases, out for 4 seconds as it decreases.")
        self.calibration_start_time = time.time()

    def acc_data_handler(self, sender, data):
        if data[0] == 0x02:
            frame_type = data[9]
            resolution = (frame_type + 1) * 8
            step = math.ceil(resolution / 8.0)
            samples = data[10:]
            n_samples = len(samples) // (step * 3)

            for i in range(n_samples):
                sample = samples[i*step*3:(i+1)*step*3]
                x = int.from_bytes(sample[0:step], byteorder='little', signed=True) / 1000.0
                y = int.from_bytes(sample[step:step*2], byteorder='little', signed=True) / 1000.0
                z = int.from_bytes(sample[step*2:step*3], byteorder='little', signed=True) / 1000.0
                
                acc = np.array([x, y, z])
                self.update_acc_vectors(acc)
                
                t = time.time()
                new_breathing_acc = self.update_breathing_acc(t)

                if new_breathing_acc:
                    if self.is_calibrating:
                        self.update_calibration(t)
                    else:
                        self.update_breathing_cycle()
                        self.update_led_count()

    def update_acc_vectors(self, acc):
        if np.isnan(self.acc_gravity).any():
            self.acc_gravity = acc
        else:
            self.acc_gravity = self.GRAVITY_ALPHA * self.acc_gravity + (1 - self.GRAVITY_ALPHA) * acc

        acc_zero_centred = acc - self.acc_gravity
        self.acc_zero_centred_exp_mean = self.ACC_MEAN_ALPHA * self.acc_zero_centred_exp_mean + (1 - self.ACC_MEAN_ALPHA) * acc_zero_centred

    def update_breathing_acc(self, t):
        if t - self.t_last_breath_acc_update > 1 / self.BR_ACC_SAMPLE_RATE:
            self.breath_acc_hist = np.roll(self.breath_acc_hist, -1)
            new_acc = np.dot(self.acc_zero_centred_exp_mean, self.acc_principle_axis)
            
            # Apply bandpass filter
            if np.all(~np.isnan(self.breath_acc_hist)):
                filtered_acc = signal.filtfilt(self.b, self.a, self.breath_acc_hist)
                new_acc = filtered_acc[-1]

            self.breath_acc_hist[-1] = new_acc
            self.breath_acc_times = np.roll(self.breath_acc_times, -1)
            self.breath_acc_times[-1] = t
            self.t_last_breath_acc_update = t

            # Check for large movements
            if not self.is_calibrating and abs(new_acc) > self.MOVEMENT_THRESHOLD:
                print("Large movement detected. Ignoring this sample.")
                return False

            return True
        return False

    def update_calibration(self, t):
        elapsed_time = t - self.calibration_start_time
        
        if elapsed_time <= self.CALIBRATION_DURATION:
            self.calibration_samples.append(self.breath_acc_hist[-1])
            
            # Calculate the target LED count for this moment in the calibration
            # Use a sawtooth wave to create a 4-second inhale, 4-second exhale pattern
            phase = (elapsed_time % self.BREATH_CYCLE_DURATION) / self.BREATH_CYCLE_DURATION
            if phase < 0.5:  # Inhale
                target_led_count = int(LED_COUNT * (2 * phase))
            else:  # Exhale
                target_led_count = int(LED_COUNT * (2 - 2 * phase))
            
            # Update the active LEDs to guide the user's breathing
            self.active_leds = target_led_count
            
            # Print the target LED count if it has changed
            if target_led_count != self.last_printed_led_count:
                breath_phase = "Inhale" if phase < 0.5 else "Exhale"
                print(f"\r{breath_phase:6s}: {'▇' * (target_led_count // 10)}{' ' * ((LED_COUNT - target_led_count) // 10)} {target_led_count:3d}", end="")
                self.last_printed_led_count = target_led_count
        else:
            self.finish_calibration()

    def finish_calibration(self):
        self.is_calibrating = False
        self.breath_min = np.min(self.calibration_samples)
        self.breath_max = np.max(self.calibration_samples)
        print("\nCalibration finished. Breathing range established.")
        print(f"Calibrated range - Min: {self.breath_min:.3f}, Max: {self.breath_max:.3f}")

    def update_breathing_cycle(self):
        current_br_phase = np.sign(self.breath_acc_hist[-1])

        if current_br_phase == self.br_last_phase or current_br_phase >= 0:
            self.br_last_phase = current_br_phase
            return

        self.current_br = 60.0 / (self.breath_acc_times[-1] - self.breath_acc_times[-2])
        
        if self.current_br > self.BR_MAX_FILTER:
            return

        self.br_last_phase = current_br_phase

    def update_led_count(self):
        if self.breath_min is not None and self.breath_max is not None:
            # Normalize the breath signal
            normalized_breath = (self.breath_acc_hist[-1] - self.breath_min) / (self.breath_max - self.breath_min)
            
            # Invert the normalized breath signal for correct LED mapping
            inverted_normalized_breath = 1 - normalized_breath
            
            # Calculate active LEDs: 150 at min (inhale), 0 at max (exhale)
            self.active_leds = int(inverted_normalized_breath * LED_COUNT)
            self.active_leds = max(0, min(LED_COUNT, self.active_leds))  # Clamp to [0, LED_COUNT]
            
            print(f"\rBreathing signal: {self.breath_acc_hist[-1]:.3f} | Active LEDs: {'▇' * (self.active_leds // 10)}{' ' * ((LED_COUNT - self.active_leds) // 10)} {self.active_leds:3d}", end="")

async def main():
    print("Scanning for Polar H10 device...")
    devices = await BleakScanner.discover()
    polar_device = next((d for d in devices if d.name and DEVICE_NAME in d.name), None)

    if not polar_device:
        print(f"{DEVICE_NAME} not found.")
        return

    print(f"Found {DEVICE_NAME}: {polar_device.address}")
    polar = PolarH10(polar_device)
    
    try:
        await polar.connect()
        await polar.start_acc_stream()
        
        # Keep the script running
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        print("Disconnecting...")
    finally:
        await polar.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
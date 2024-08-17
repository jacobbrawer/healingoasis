import asyncio
import time
import numpy as np
import math
from bleak import BleakClient, BleakScanner
from scipy import signal
import board
import neopixel_spi as neopixel

# Polar H10 UUIDs
PMD_SERVICE = "fb005c80-02e7-f387-1cad-8acd2d8df0c8"
PMD_CONTROL = "fb005c81-02e7-f387-1cad-8acd2d8df0c8"
PMD_DATA = "fb005c82-02e7-f387-1cad-8acd2d8df0c8"
HR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"
DEVICE_NAME = "Polar H10"

# LED configuration
LED_COUNT = 150
LED_BRIGHTNESS = 0.5
SPI_DEVICE = board.SPI()

class PolarH10:
    def __init__(self):
        self.device = None
        self.client = None
        self.active_leds = 0
        self.pixels = neopixel.NeoPixel_SPI(SPI_DEVICE, LED_COUNT, brightness=LED_BRIGHTNESS, auto_write=False, pixel_order=neopixel.GRB)

        # Colors
        self.BLUE = (0, 0, 255)
        self.ORANGE = (255, 165, 0)
        self.OFF = (0, 0, 0)
        self.RED = (255, 0, 0)

        # Heart rate parameters
        self.heart_rate = 0
        self.heart_rate_detected = False
        self.heart_rate_detection_start = None
        self.HEART_RATE_DETECTION_DURATION = 5  # seconds

        # Accelerometer parameters
        self.ACC_SAMPLE_RATE = 200  # Hz
        self.GRAVITY_ALPHA = 0.999
        self.ACC_MEAN_ALPHA = 0.98
        self.acc_principle_axis = np.array([0, 0, 1])

        # Breathing signal parameters
        self.BR_ACC_SAMPLE_RATE = 10  # Hz
        self.BR_ACC_HIST_SIZE = 1200
        self.BR_MAX_FILTER = 30
        self.MOVEMENT_THRESHOLD = 0.1

        # Calibration parameters
        self.CALIBRATION_DURATION = 32  # 32 seconds for 4 full breath cycles
        self.BREATH_CYCLE_DURATION = 8
        self.is_calibrating = False
        self.calibration_start_time = None
        self.calibration_samples = []
        self.breath_min = None
        self.breath_max = None
        self.last_active_leds = -1
        self.calibration_completed = False

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

        # Debug counters
        self.sample_count = 0
        self.last_print_time = time.time()

    async def connect(self):
        while True:
            try:
                print("Scanning for Polar H10 device...")
                devices = await BleakScanner.discover()
                self.device = next((d for d in devices if d.name and DEVICE_NAME in d.name), None)

                if not self.device:
                    print(f"{DEVICE_NAME} not found. Retrying in 10 seconds...")
                    await asyncio.sleep(10)
                    continue

                print(f"Found {DEVICE_NAME}: {self.device.address}")
                self.client = BleakClient(self.device)
                await self.client.connect(timeout=30.0)
                print(f"Connected: {self.client.is_connected}")

                await self.start_heart_rate_stream()
                await self.start_acc_stream()
                return True
            except Exception as e:
                print(f"Connection failed: {str(e)}. Retrying in 10 seconds...")
                await asyncio.sleep(10)

    async def disconnect(self):
        if self.client and self.client.is_connected:
            await self.client.disconnect()
        self.pixels.fill(self.OFF)
        self.pixels.show()

    async def start_heart_rate_stream(self):
        await self.client.start_notify(HR_UUID, self.heart_rate_handler)
        print("Heart rate stream started")

    def heart_rate_handler(self, sender, data):
        self.heart_rate = data[1]
        if self.heart_rate > 0:
            if not self.heart_rate_detected:
                self.heart_rate_detected = True
                self.heart_rate_detection_start = time.time()
                print("Heart rate detected. Starting 5-second countdown.")
            elif not self.is_calibrating and not self.calibration_completed and time.time() - self.heart_rate_detection_start >= self.HEART_RATE_DETECTION_DURATION:
                asyncio.create_task(self.start_calibration())
        else:
            self.heart_rate_detected = False
            self.heart_rate_detection_start = None
        print(f"Heart rate: {self.heart_rate} BPM")

    async def start_acc_stream(self):
        await self.client.write_gatt_char(PMD_CONTROL, bytearray([0x02, 0x02, 0x00, 0x01, 0xC8, 0x00, 0x01, 0x01, 0x10, 0x00, 0x02, 0x01, 0x08, 0x00]), response=True)
        await self.client.start_notify(PMD_DATA, self.acc_data_handler)
        print("ACC stream started")
        self.pixels.fill(self.OFF)
        self.pixels.show()

    def acc_data_handler(self, sender, data):
        try:
            if data[0] == 0x02:
                frame_type = data[9]
                resolution = (frame_type + 1) * 8
                step = math.ceil(resolution / 8.0)
                samples = data[10:]
                n_samples = len(samples) // (step * 3)

                self.sample_count += n_samples

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
                        elif self.calibration_completed:
                            self.update_breathing_cycle()
                            self.update_led_count()

                # Print debug info every 5 seconds
                current_time = time.time()
                if current_time - self.last_print_time > 5:
                    print(f"Processed {self.sample_count} samples in the last 5 seconds")
                    print(f"Last acceleration: x={x:.3f}, y={y:.3f}, z={z:.3f}")
                    print(f"Heart rate detected: {self.heart_rate_detected}")
                    print(f"Is calibrating: {self.is_calibrating}")
                    print(f"Calibration completed: {self.calibration_completed}")
                    print(f"Breath acc: {self.breath_acc_hist[-1]:.3f}")
                    if self.heart_rate_detected and not self.is_calibrating and not self.calibration_completed:
                        time_left = max(0, self.HEART_RATE_DETECTION_DURATION - (current_time - self.heart_rate_detection_start))
                        print(f"Time until calibration starts: {time_left:.1f} seconds")
                    self.sample_count = 0
                    self.last_print_time = current_time

        except Exception as e:
            print(f"Error in acc_data_handler: {str(e)}")

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
            
            if np.all(~np.isnan(self.breath_acc_hist)):
                filtered_acc = signal.filtfilt(self.b, self.a, self.breath_acc_hist)
                new_acc = filtered_acc[-1]

            self.breath_acc_hist[-1] = new_acc
            self.breath_acc_times = np.roll(self.breath_acc_times, -1)
            self.breath_acc_times[-1] = t
            self.t_last_breath_acc_update = t

            if not self.is_calibrating and abs(new_acc) > self.MOVEMENT_THRESHOLD:
                print("Large movement detected. Ignoring this sample.")
                return False

            return True
        return False

    async def start_calibration(self):
        # Flash red to notify calibration is starting
        await self.flash_notification()

        self.is_calibrating = True
        self.calibration_start_time = time.time()
        self.calibration_samples = []
        print(f"Calibration started. Follow the breath guide for {self.CALIBRATION_DURATION} seconds.")
        print("Breathe in as the blue lights increase, out as they decrease.")
        
        # Start the calibration process immediately
        asyncio.create_task(self.run_calibration())

    async def flash_notification(self):
        self.update_leds(color=self.RED, num_leds=LED_COUNT)
        await asyncio.sleep(0.5)
        self.update_leds(color=self.OFF, num_leds=LED_COUNT)
        await asyncio.sleep(0.5)

    async def run_calibration(self):
        start_time = time.time()
        while time.time() - start_time < self.CALIBRATION_DURATION:
            elapsed_time = time.time() - start_time
            phase = (elapsed_time % self.BREATH_CYCLE_DURATION) / self.BREATH_CYCLE_DURATION
            if phase < 0.5:  # Inhale
                target_led_count = int(LED_COUNT * (2 * phase))
            else:  # Exhale
                target_led_count = int(LED_COUNT * (2 - 2 * phase))
            
            self.update_leds(color=self.BLUE, num_leds=target_led_count)  # Blue color for calibration
            await asyncio.sleep(0.05)  # Update LEDs every 50ms for smooth transition
        
        self.finish_calibration()

    def update_calibration(self, t):
        if self.is_calibrating:
            self.calibration_samples.append(self.breath_acc_hist[-1])

    def finish_calibration(self):
        if self.calibration_samples:
            self.is_calibrating = False
            self.calibration_completed = True
            self.breath_min = np.min(self.calibration_samples)
            self.breath_max = np.max(self.calibration_samples)
            print("\nCalibration finished. Breathing range established.")
            print(f"Calibrated range - Min: {self.breath_min:.3f}, Max: {self.breath_max:.3f}")
            self.pixels.fill(self.OFF)
            self.pixels.show()
        else:
            print("\nCalibration failed: No samples collected.")
            self.is_calibrating = False
            self.calibration_completed = False
            self.heart_rate_detected = False  # Reset to allow for recalibration

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
            normalized_breath = (self.breath_acc_hist[-1] - self.breath_min) / (self.breath_max - self.breath_min)
            inverted_normalized_breath = 1 - normalized_breath
            self.active_leds = int(inverted_normalized_breath * LED_COUNT)
            self.active_leds = max(0, min(LED_COUNT, self.active_leds))
            self.update_leds(color=self.ORANGE, num_leds=self.active_leds)  # Orange color for normal operation

    def update_leds(self, color, num_leds):
        try:
            self.pixels.fill(self.OFF)  # Turn off all LEDs
            for i in range(num_leds):
                self.pixels[i] = color
            self.pixels.show()
        except Exception as e:
           		print(f"Error updating LEDs: {str(e)}")

async def main():
    polar = PolarH10()
    
    while True:
        if not polar.client or not polar.client.is_connected:
            await polar.connect()
        
        try:
            while polar.client.is_connected:
                await asyncio.sleep(1)  # Check connection more frequently
        except Exception as e:
            print(f"Error: {str(e)}. Attempting to reconnect...")
        
        await polar.disconnect()

if __name__ == "__main__":
    asyncio.run(main())

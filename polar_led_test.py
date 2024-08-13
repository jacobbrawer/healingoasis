import asyncio
import time
import numpy as np
import math
from bleak import BleakClient, BleakScanner
import board
import neopixel

# Polar H10 UUIDs
PMD_SERVICE = "fb005c80-02e7-f387-1cad-8acd2d8df0c8"
PMD_CONTROL = "fb005c81-02e7-f387-1cad-8acd2d8df0c8"
PMD_DATA = "fb005c82-02e7-f387-1cad-8acd2d8df0c8"
DEVICE_NAME = "Polar H10"

# LED strip configuration:
LED_COUNT = 60        # Number of LED pixels.
LED_PIN = board.D18   # GPIO pin connected to the pixels (18 uses PWM!)
LED_BRIGHTNESS = 0.5  # Set to 0 for darkest and 1 for brightest

# Create NeoPixel object with appropriate configuration
pixels = neopixel.NeoPixel(LED_PIN, LED_COUNT, brightness=LED_BRIGHTNESS, auto_write=False)

class PolarH10:
    def __init__(self, device):
        self.device = device
        self.client = None
        self.led_brightness = 0
        self.acc_z_smooth = 0
        self.smoothing_factor = 0.1  # Adjust this for more or less smoothing
        self.max_z = -float('inf')  # Maximum expected Z acceleration
        self.min_z = float('inf')  # Minimum expected Z acceleration

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

    def acc_data_handler(self, sender, data):
        if data[0] == 0x02:
            frame_type = data[9]
            resolution = (frame_type + 1) * 8
            step = math.ceil(resolution / 8.0)
            samples = data[10:]
            n_samples = len(samples) // (step * 3)

            for i in range(n_samples):
                sample = samples[i*step*3:(i+1)*step*3]
                z = int.from_bytes(sample[step*2:step*3], byteorder='little', signed=True) / 1000.0
                
                # Smooth the Z-axis acceleration
                self.acc_z_smooth = self.acc_z_smooth * (1 - self.smoothing_factor) + z * self.smoothing_factor

                # Update min and max Z values
                self.min_z = min(self.min_z, self.acc_z_smooth)
                self.max_z = max(self.max_z, self.acc_z_smooth)

                # Map smoothed Z-axis acceleration to LED brightness (0-255)
                if self.max_z != self.min_z:
                    normalized_z = (self.max_z - self.acc_z_smooth) / (self.max_z - self.min_z)
                    self.led_brightness = int(normalized_z * 255)
                else:
                    self.led_brightness = 127  # Default to middle brightness if max == min

                self.led_brightness = max(0, min(255, self.led_brightness))  # Clamp to [0, 255]

                # Update LED strip
                pixels.fill((self.led_brightness, self.led_brightness, self.led_brightness))
                pixels.show()

                print(f"Z-axis acceleration: {z:.3f}")
                print(f"Smoothed Z-axis: {self.acc_z_smooth:.3f}")
                print(f"Min Z: {self.min_z:.3f}, Max Z: {self.max_z:.3f}")
                print(f"LED Brightness: {self.led_brightness}")

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
        # Turn off all LEDs
        pixels.fill((0, 0, 0))
        pixels.show()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Script terminated by user")
    finally:
        # Turn off all LEDs
        pixels.fill((0, 0, 0))
        pixels.show()
"""
Simulated IMU Data Generator
============================

This module simulates Arduino IMU sensor output for testing the rehabilitation
system without physical hardware. Useful for:
- Software development
- Testing algorithms
- Demonstrations
- Training

Generates realistic bicep curl motion data.
"""

import time
import math
import random
import threading
import queue
from dataclasses import dataclass
import numpy as np


@dataclass
class SimulatedIMUReading:
    """Simulated IMU reading matching Arduino format."""
    timestamp: float
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float


class BicepCurlSimulator:
    """
    Simulates realistic bicep curl IMU sensor data.
    
    Models:
    - Smooth sinusoidal motion
    - Gravity effects on accelerometer
    - Gyroscope angular velocity
    - Realistic noise
    - Variable speed and range
    """
    
    def __init__(
        self,
        movement_speed: float = 1.0,
        angle_range: tuple = (0, 140),
        noise_level: float = 0.05,
        perfect_technique: bool = True
    ):
        """
        Initialize bicep curl simulator.
        
        Args:
            movement_speed: Speed multiplier (1.0 = normal, 2.0 = fast)
            angle_range: (min_angle, max_angle) in degrees
            noise_level: Sensor noise magnitude (0 = perfect, 1 = very noisy)
            perfect_technique: If False, adds technique errors
        """
        self.movement_speed = movement_speed
        self.min_angle = angle_range[0]
        self.max_angle = angle_range[1]
        self.noise_level = noise_level
        self.perfect_technique = perfect_technique
        
        # State
        self.current_time = 0.0
        self.phase_offset = 0.0
        
        # Add imperfections if not perfect technique
        if not perfect_technique:
            # Random technique errors
            self.speed_variation = random.uniform(0.8, 1.2)
            self.range_reduction = random.uniform(0.7, 0.95)
            self.jerkiness = random.uniform(0.1, 0.3)
        else:
            self.speed_variation = 1.0
            self.range_reduction = 1.0
            self.jerkiness = 0.0
    
    def get_reading(self) -> SimulatedIMUReading:
        """
        Generate next simulated IMU reading.
        
        Returns:
            SimulatedIMUReading with realistic sensor data
        """
        # Calculate current angle using sinusoidal motion
        # One complete bicep curl cycle
        frequency = 0.5 * self.movement_speed * self.speed_variation  # Hz
        
        # Add jerkiness by modulating frequency
        if random.random() < self.jerkiness:
            frequency *= random.uniform(0.7, 1.3)
        
        t = self.current_time / 1000.0  # Convert to seconds
        
        # Sinusoidal angle (0 to max_angle)
        angle_amplitude = (self.max_angle - self.min_angle) / 2 * self.range_reduction
        angle_center = self.min_angle + angle_amplitude
        
        angle_deg = angle_center + angle_amplitude * math.sin(2 * math.pi * frequency * t + self.phase_offset)
        angle_rad = math.radians(angle_deg)
        
        # Calculate angular velocity (derivative of angle)
        angular_velocity_rad = 2 * math.pi * frequency * angle_amplitude * math.cos(2 * math.pi * frequency * t + self.phase_offset)
        angular_velocity_deg = math.degrees(angular_velocity_rad)
        
        # Simulate accelerometer (measures gravity + motion)
        # For a rotating forearm, acceleration components depend on angle
        g = 9.81  # m/s²
        
        # Gravity component in sensor frame
        ax = g * math.sin(angle_rad) * 0.1  # Small lateral component
        ay = g * math.sin(angle_rad)        # Main rotation axis
        az = g * math.cos(angle_rad)        # Vertical component
        
        # Add motion acceleration (centripetal + tangential)
        motion_accel = 0.2 * abs(angular_velocity_rad)  # Simplified
        ay += motion_accel * math.sin(angle_rad)
        az += motion_accel * math.cos(angle_rad)
        
        # Simulate gyroscope (angular velocity in sensor frame)
        # Primary rotation around Y axis for bicep curl
        gx = angular_velocity_deg * 0.1  # Small x component
        gy = angular_velocity_deg        # Main rotation
        gz = angular_velocity_deg * 0.05 # Small z component
        
        # Add realistic sensor noise
        ax += random.gauss(0, self.noise_level * g)
        ay += random.gauss(0, self.noise_level * g)
        az += random.gauss(0, self.noise_level * g)
        gx += random.gauss(0, self.noise_level * 10)
        gy += random.gauss(0, self.noise_level * 10)
        gz += random.gauss(0, self.noise_level * 10)
        
        # Create reading
        reading = SimulatedIMUReading(
            timestamp=self.current_time,
            ax=ax,
            ay=ay,
            az=az,
            gx=gx,
            gy=gy,
            gz=gz
        )
        
        # Advance time (10ms per reading = 100 Hz)
        self.current_time += 10.0
        
        return reading


class SimulatedSerialPort:
    """
    Simulates Arduino serial port for testing.
    
    Drop-in replacement for pyserial Serial class.
    Generates realistic IMU data without hardware.
    """
    
    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        timeout: float = 1.0,
        simulator_params: dict = None
    ):
        """
        Initialize simulated serial port.
        
        Args:
            port: Ignored (for compatibility)
            baudrate: Ignored (for compatibility)
            timeout: Read timeout
            simulator_params: Parameters for BicepCurlSimulator
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.is_open = True
        
        # Initialize simulator
        params = simulator_params or {}
        self.simulator = BicepCurlSimulator(**params)
        
        # Buffer for readline()
        self.line_buffer = queue.Queue()
        
        # Background thread to generate data
        self.running = False
        self.generator_thread = None
        
        print(f"[SIMULATION MODE] Simulated serial port on '{port}'")
    
    def _data_generator(self):
        """Background thread that generates IMU readings."""
        while self.running:
            reading = self.simulator.get_reading()
            
            # Format as CSV line (matching Arduino output)
            line = (
                f"{reading.timestamp:.0f},"
                f"{reading.ax:.4f},"
                f"{reading.ay:.4f},"
                f"{reading.az:.4f},"
                f"{reading.gx:.4f},"
                f"{reading.gy:.4f},"
                f"{reading.gz:.4f}\n"
            )
            
            self.line_buffer.put(line)
            
            # Simulate 100 Hz sampling rate
            time.sleep(0.01)
    
    def readline(self) -> bytes:
        """
        Read one line of CSV data.
        
        Returns:
            Encoded CSV line
        """
        if not self.running:
            # Start generator on first read
            self.running = True
            self.generator_thread = threading.Thread(target=self._data_generator, daemon=True)
            self.generator_thread.start()
        
        try:
            line = self.line_buffer.get(timeout=self.timeout)
            return line.encode('utf-8')
        except queue.Empty:
            return b''
    
    def reset_input_buffer(self):
        """Clear input buffer."""
        while not self.line_buffer.empty():
            try:
                self.line_buffer.get_nowait()
            except queue.Empty:
                break
    
    def close(self):
        """Close the simulated port."""
        self.running = False
        self.is_open = False
        if self.generator_thread:
            self.generator_thread.join(timeout=1.0)
        print("[SIMULATION MODE] Simulated port closed")


def create_test_scenarios():
    """
    Create different test scenarios for development.
    
    Returns:
        Dictionary of scenario name -> simulator parameters
    """
    return {
        'perfect_technique': {
            'movement_speed': 1.0,
            'angle_range': (10, 150),
            'noise_level': 0.02,
            'perfect_technique': True
        },
        'too_fast': {
            'movement_speed': 2.5,
            'angle_range': (10, 150),
            'noise_level': 0.05,
            'perfect_technique': False
        },
        'too_slow': {
            'movement_speed': 0.4,
            'angle_range': (10, 150),
            'noise_level': 0.03,
            'perfect_technique': False
        },
        'insufficient_range': {
            'movement_speed': 1.0,
            'angle_range': (20, 100),
            'noise_level': 0.04,
            'perfect_technique': False
        },
        'jerky_movement': {
            'movement_speed': 1.0,
            'angle_range': (10, 150),
            'noise_level': 0.08,
            'perfect_technique': False
        },
        'noisy_sensor': {
            'movement_speed': 1.0,
            'angle_range': (10, 150),
            'noise_level': 0.15,
            'perfect_technique': True
        }
    }


# Patch for easy testing
def patch_serial_for_simulation(scenario: str = 'perfect_technique'):
    """
    Monkey-patch pyserial to use simulation.
    
    Usage:
        patch_serial_for_simulation('too_fast')
        # Now all Serial() calls use simulated data
    
    Args:
        scenario: Test scenario name
    """
    import serial as serial_module
    
    scenarios = create_test_scenarios()
    
    if scenario not in scenarios:
        print(f"Unknown scenario '{scenario}'. Available: {list(scenarios.keys())}")
        return
    
    params = scenarios[scenario]
    
    # Store original Serial class
    if not hasattr(serial_module, '_original_Serial'):
        serial_module._original_Serial = serial_module.Serial
    
    # Replace with simulated version
    def simulated_serial_factory(port, baudrate=115200, timeout=1.0, **kwargs):
        return SimulatedSerialPort(port, baudrate, timeout, simulator_params=params)
    
    serial_module.Serial = simulated_serial_factory
    
    print(f"\n{'='*60}")
    print(f"SIMULATION MODE ENABLED: '{scenario}'")
    print(f"{'='*60}")
    print(f"Parameters: {params}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    """
    Test the simulator directly.
    """
    print("IMU Data Simulator Test")
    print("=" * 60)
    
    # Create simulators for different scenarios
    scenarios = create_test_scenarios()
    
    for scenario_name, params in scenarios.items():
        print(f"\nScenario: {scenario_name}")
        print(f"Parameters: {params}")
        
        simulator = BicepCurlSimulator(**params)
        
        # Generate 10 readings
        print("Sample readings:")
        for i in range(10):
            reading = simulator.get_reading()
            print(f"  {reading.timestamp:.0f}, {reading.ax:.2f}, {reading.ay:.2f}, {reading.az:.2f}, "
                  f"{reading.gx:.2f}, {reading.gy:.2f}, {reading.gz:.2f}")
            time.sleep(0.01)
        
        print()
    
    # Test simulated serial port
    print("\nTesting Simulated Serial Port")
    print("=" * 60)
    
    port = SimulatedSerialPort('/dev/null', simulator_params=scenarios['perfect_technique'])
    
    print("Reading 20 lines:")
    for i in range(20):
        line = port.readline().decode('utf-8').strip()
        print(f"  {line}")
    
    port.close()
    
    print("\n✓ Simulator test complete!")
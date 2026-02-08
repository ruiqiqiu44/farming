"""
IMU Data Processor for Rehabilitation System
============================================

This module handles real-time IMU data processing from Arduino sensors,
including data acquisition, filtering, motion estimation, and reference comparison.

Author: Rehabilitation Systems Engineering Team
Purpose: Enable real-time motion feedback for bicep curl rehabilitation
"""

import serial
import numpy as np
from collections import deque
from scipy import signal
from scipy.spatial.transform import Rotation
import threading
import queue
import time
import json
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict
import logging

# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class IMUReading:
    """
    Represents a single timestamped IMU reading.
    
    Attributes:
        timestamp: Time in milliseconds from Arduino
        ax, ay, az: Accelerometer readings (m/s²)
        gx, gy, gz: Gyroscope readings (rad/s or deg/s depending on sensor config)
    """
    timestamp: float
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float


@dataclass
class MotionState:
    """
    Processed motion state derived from IMU data.
    
    This contains higher-level motion information suitable for
    rehabilitation assessment and visualization.
    
    Attributes:
        timestamp: Time of measurement
        elbow_angle: Estimated elbow flexion angle (degrees, 0=extended, 180=fully flexed)
        angular_velocity: Rate of angle change (deg/s)
        position_2d: 2D position for visualization (x, y in pixels or normalized coords)
        position_3d: 3D position estimate (x, y, z)
        acceleration_magnitude: Total acceleration magnitude
        movement_phase: Current phase (e.g., 'flexion', 'extension', 'hold')
    """
    timestamp: float
    elbow_angle: float
    angular_velocity: float
    position_2d: Tuple[float, float]
    position_3d: Tuple[float, float, float]
    acceleration_magnitude: float
    movement_phase: str


@dataclass
class ComparisonResult:
    """
    Results of comparing patient motion to reference (doctor) motion.
    
    Attributes:
        similarity_score: Overall similarity (0-100, higher is better)
        angle_error: RMS error in joint angle (degrees)
        timing_ratio: Patient speed / Reference speed
        smoothness_score: Movement smoothness (0-100, higher is smoother)
        error_type: Descriptive error message
        recommendations: List of improvement suggestions
    """
    similarity_score: float
    angle_error: float
    timing_ratio: float
    smoothness_score: float
    error_type: str
    recommendations: List[str]


class LowPassFilter:
    """
    Butterworth low-pass filter for IMU noise reduction.
    
    Used to smooth accelerometer and gyroscope readings while maintaining
    response time suitable for real-time rehabilitation feedback.
    """
    
    def __init__(self, cutoff_freq: float = 5.0, sampling_rate: float = 100.0, order: int = 2):
        """
        Initialize the low-pass filter.
        
        Args:
            cutoff_freq: Cutoff frequency in Hz (typical human motion: 1-10 Hz)
            sampling_rate: IMU sampling rate in Hz
            order: Filter order (higher = steeper rolloff but more delay)
        """
        self.cutoff_freq = cutoff_freq
        self.sampling_rate = sampling_rate
        self.order = order
        
        # Design the Butterworth filter
        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff_freq / nyquist
        self.b, self.a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        
        # Store filter state for continuous filtering
        self.z = signal.lfilter_zi(self.b, self.a)
    
    def filter(self, data_point: float) -> float:
        """
        Apply filter to a single data point.
        
        Args:
            data_point: New sensor reading
            
        Returns:
            Filtered value
        """
        filtered, self.z = signal.lfilter(self.b, self.a, [data_point], zi=self.z)
        return filtered[0]
    
    def reset(self):
        """Reset filter state (use when starting new recording)."""
        self.z = signal.lfilter_zi(self.b, self.a)


class IMUSerialReader(threading.Thread):
    """
    Non-blocking serial reader for Arduino IMU data.
    
    Runs in a separate thread to continuously read CSV data from Arduino
    without blocking the main application or game loop.
    """
    
    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 1.0):
        """
        Initialize serial connection to Arduino.
        
        Args:
            port: Serial port (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux)
            baudrate: Communication speed (must match Arduino configuration)
            timeout: Read timeout in seconds
        """
        super().__init__(daemon=True)  # Daemon thread exits when main program exits
        
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        
        # Thread-safe queue for passing data to main application
        self.data_queue = queue.Queue(maxsize=1000)
        
        # Control flags
        self.running = False
        self.connected = False
        
        # Serial connection (initialized in run())
        self.serial_conn = None
        
        logger.info(f"IMU Serial Reader initialized for port {port}")
    
    def run(self):
        """
        Main thread loop - continuously reads serial data.
        
        This method runs in a separate thread and should not be called directly.
        Use start() to begin reading.
        """
        self.running = True
        
        try:
            # Establish serial connection
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            self.connected = True
            logger.info(f"Connected to Arduino on {self.port}")
            
            # Allow Arduino to reset and stabilize
            time.sleep(2)
            self.serial_conn.reset_input_buffer()
            
            while self.running:
                try:
                    # Read one line of CSV data
                    line = self.serial_conn.readline().decode('utf-8').strip()
                    
                    if not line:
                        continue
                    
                    # Parse CSV: timestamp, ax, ay, az, gx, gy, gz
                    parts = line.split(',')
                    
                    if len(parts) != 7:
                        logger.warning(f"Invalid data format: {line}")
                        continue
                    
                    # Convert to IMUReading object
                    reading = IMUReading(
                        timestamp=float(parts[0]),
                        ax=float(parts[1]),
                        ay=float(parts[2]),
                        az=float(parts[3]),
                        gx=float(parts[4]),
                        gy=float(parts[5]),
                        gz=float(parts[6])
                    )
                    
                    # Add to queue (non-blocking: if queue full, discard oldest)
                    try:
                        self.data_queue.put_nowait(reading)
                    except queue.Full:
                        # Remove oldest reading and add new one
                        try:
                            self.data_queue.get_nowait()
                            self.data_queue.put_nowait(reading)
                        except queue.Empty:
                            pass
                
                except UnicodeDecodeError:
                    logger.warning("Failed to decode serial data")
                except ValueError as e:
                    logger.warning(f"Failed to parse IMU data: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error in serial reading: {e}")
        
        except serial.SerialException as e:
            logger.error(f"Serial connection error: {e}")
            self.connected = False
        
        finally:
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()
                logger.info("Serial connection closed")
    
    def stop(self):
        """Stop the reading thread gracefully."""
        self.running = False
        logger.info("Stopping IMU serial reader")
    
    def get_reading(self, timeout: float = 0.1) -> Optional[IMUReading]:
        """
        Get the next IMU reading from the queue.
        
        Args:
            timeout: Maximum time to wait for data
            
        Returns:
            IMUReading object or None if no data available
        """
        try:
            return self.data_queue.get(timeout=timeout)
        except queue.Empty:
            return None


class MotionProcessor:
    """
    Processes raw IMU data into meaningful motion metrics for rehabilitation.
    
    This class handles:
    - Data filtering
    - Joint angle estimation
    - Movement phase detection
    - Position calculation for visualization
    """
    
    def __init__(self, sampling_rate: float = 100.0):
        """
        Initialize motion processor.
        
        Args:
            sampling_rate: Expected IMU sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        
        # Initialize filters for each axis
        self.filters = {
            'ax': LowPassFilter(cutoff_freq=5.0, sampling_rate=sampling_rate),
            'ay': LowPassFilter(cutoff_freq=5.0, sampling_rate=sampling_rate),
            'az': LowPassFilter(cutoff_freq=5.0, sampling_rate=sampling_rate),
            'gx': LowPassFilter(cutoff_freq=5.0, sampling_rate=sampling_rate),
            'gy': LowPassFilter(cutoff_freq=5.0, sampling_rate=sampling_rate),
            'gz': LowPassFilter(cutoff_freq=5.0, sampling_rate=sampling_rate),
        }
        
        # State tracking
        self.previous_angle = 0.0
        self.previous_timestamp = 0.0
        
        # Movement buffer for phase detection
        self.angle_buffer = deque(maxlen=20)  # Last 20 angle readings
        
        # Constants for bicep curl kinematics
        self.FOREARM_LENGTH = 0.35  # meters (approximate, can be calibrated)
        self.UPPER_ARM_LENGTH = 0.30  # meters
        
        logger.info("Motion processor initialized")
    
    def process_reading(self, raw_reading: IMUReading) -> MotionState:
        """
        Convert raw IMU reading to processed motion state.
        
        Args:
            raw_reading: Raw IMU data from sensor
            
        Returns:
            MotionState with derived metrics
        """
        # Step 1: Apply low-pass filtering to reduce noise
        filtered_ax = self.filters['ax'].filter(raw_reading.ax)
        filtered_ay = self.filters['ay'].filter(raw_reading.ay)
        filtered_az = self.filters['az'].filter(raw_reading.az)
        filtered_gx = self.filters['gx'].filter(raw_reading.gx)
        filtered_gy = self.filters['gy'].filter(raw_reading.gy)
        filtered_gz = self.filters['gz'].filter(raw_reading.gz)
        
        # Step 2: Estimate elbow angle from accelerometer orientation
        # For bicep curl, we primarily use the Y-Z plane
        # This is a simplified model - more sophisticated fusion can be added
        elbow_angle = self._estimate_elbow_angle(filtered_ay, filtered_az)
        
        # Step 3: Calculate angular velocity
        dt = (raw_reading.timestamp - self.previous_timestamp) / 1000.0  # Convert ms to s
        if dt > 0:
            angular_velocity = (elbow_angle - self.previous_angle) / dt
        else:
            angular_velocity = 0.0
        
        # Step 4: Detect movement phase
        self.angle_buffer.append(elbow_angle)
        movement_phase = self._detect_movement_phase()
        
        # Step 5: Calculate visualization positions
        position_2d = self._calculate_2d_position(elbow_angle)
        position_3d = self._calculate_3d_position(elbow_angle, filtered_ax, filtered_ay, filtered_az)
        
        # Step 6: Calculate acceleration magnitude
        acc_magnitude = np.sqrt(filtered_ax**2 + filtered_ay**2 + filtered_az**2)
        
        # Update state
        self.previous_angle = elbow_angle
        self.previous_timestamp = raw_reading.timestamp
        
        return MotionState(
            timestamp=raw_reading.timestamp,
            elbow_angle=elbow_angle,
            angular_velocity=angular_velocity,
            position_2d=position_2d,
            position_3d=position_3d,
            acceleration_magnitude=acc_magnitude,
            movement_phase=movement_phase
        )
    
    def _estimate_elbow_angle(self, ay: float, az: float) -> float:
        """
        Estimate elbow flexion angle from accelerometer data.
        
        For a bicep curl, the forearm rotates primarily in the sagittal plane.
        We use the accelerometer to measure the angle of the forearm relative to gravity.
        
        Args:
            ay, az: Filtered accelerometer components
            
        Returns:
            Elbow angle in degrees (0 = extended, 180 = fully flexed)
        """
        # Calculate angle from vertical using arctangent
        # Note: This assumes sensor is mounted with specific orientation
        angle_rad = np.arctan2(ay, az)
        angle_deg = np.degrees(angle_rad)
        
        # Map to 0-180 range for elbow flexion
        # Add 90 to shift range and ensure positive values
        elbow_angle = (angle_deg + 90) % 180
        
        # Clamp to physiological range for bicep curl
        elbow_angle = np.clip(elbow_angle, 0, 160)
        
        return elbow_angle
    
    def _detect_movement_phase(self) -> str:
        """
        Detect current phase of bicep curl movement.
        
        Phases:
        - 'flexion': Elbow angle increasing (lifting)
        - 'extension': Elbow angle decreasing (lowering)
        - 'hold': Minimal movement
        
        Returns:
            Movement phase as string
        """
        if len(self.angle_buffer) < 5:
            return 'hold'
        
        # Calculate trend over recent angles
        recent_angles = list(self.angle_buffer)[-5:]
        angle_change = recent_angles[-1] - recent_angles[0]
        
        # Threshold for detecting movement vs hold
        MOVEMENT_THRESHOLD = 3.0  # degrees
        
        if angle_change > MOVEMENT_THRESHOLD:
            return 'flexion'
        elif angle_change < -MOVEMENT_THRESHOLD:
            return 'extension'
        else:
            return 'hold'
    
    def _calculate_2d_position(self, elbow_angle: float) -> Tuple[float, float]:
        """
        Calculate 2D screen position for visualization.
        
        Maps elbow angle to a circular arc representing forearm position.
        Suitable for displaying as a moving circle in game interface.
        
        Args:
            elbow_angle: Current elbow angle in degrees
            
        Returns:
            (x, y) position in normalized coordinates (-1 to 1)
        """
        # Convert angle to radians
        angle_rad = np.radians(elbow_angle)
        
        # Calculate position on circular arc
        # Assumes shoulder is at origin, forearm sweeps in arc
        x = self.FOREARM_LENGTH * np.sin(angle_rad)
        y = -self.FOREARM_LENGTH * np.cos(angle_rad)  # Negative for screen coordinates
        
        # Normalize to -1 to 1 range for easy scaling
        max_reach = self.FOREARM_LENGTH
        x_norm = x / max_reach
        y_norm = y / max_reach
        
        return (x_norm, y_norm)
    
    def _calculate_3d_position(self, elbow_angle: float, ax: float, ay: float, az: float) -> Tuple[float, float, float]:
        """
        Calculate 3D position estimate for advanced visualization.
        
        Args:
            elbow_angle: Current elbow angle
            ax, ay, az: Accelerometer readings
            
        Returns:
            (x, y, z) position in meters
        """
        # Simplified 3D model - assumes forearm in Y-Z plane
        angle_rad = np.radians(elbow_angle)
        
        x = 0.0  # No lateral movement in basic bicep curl
        y = self.FOREARM_LENGTH * np.sin(angle_rad)
        z = self.FOREARM_LENGTH * np.cos(angle_rad)
        
        return (x, y, z)
    
    def reset(self):
        """Reset processor state (use when starting new recording)."""
        for f in self.filters.values():
            f.reset()
        self.previous_angle = 0.0
        self.previous_timestamp = 0.0
        self.angle_buffer.clear()
        logger.info("Motion processor reset")


class ReferenceComparator:
    """
    Compares patient motion against reference (doctor) motion.
    
    Uses rule-based assessment to provide explainable feedback
    suitable for rehabilitation applications.
    """
    
    def __init__(self):
        """Initialize the comparator."""
        self.reference_motion: List[MotionState] = []
        self.reference_loaded = False
        
        # Analysis parameters
        self.angle_tolerance = 15.0  # degrees
        self.timing_tolerance = 0.3  # 30% speed variation acceptable
        
        logger.info("Reference comparator initialized")
    
    def load_reference(self, reference_data: List[MotionState]):
        """
        Load reference motion from doctor's performance.
        
        Args:
            reference_data: List of MotionState objects from doctor's recording
        """
        self.reference_motion = reference_data
        self.reference_loaded = True
        logger.info(f"Loaded reference motion with {len(reference_data)} frames")
    
    def save_reference(self, filepath: str):
        """
        Save reference motion to file.
        
        Args:
            filepath: Path to JSON file
        """
        if not self.reference_loaded:
            logger.warning("No reference motion to save")
            return
        
        data = [asdict(state) for state in self.reference_motion]
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Reference motion saved to {filepath}")
    
    def load_reference_from_file(self, filepath: str):
        """
        Load reference motion from file.
        
        Args:
            filepath: Path to JSON file
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.reference_motion = [
            MotionState(
                timestamp=d['timestamp'],
                elbow_angle=d['elbow_angle'],
                angular_velocity=d['angular_velocity'],
                position_2d=tuple(d['position_2d']),
                position_3d=tuple(d['position_3d']),
                acceleration_magnitude=d['acceleration_magnitude'],
                movement_phase=d['movement_phase']
            )
            for d in data
        ]
        
        self.reference_loaded = True
        logger.info(f"Loaded reference motion from {filepath}")
    
    def compare_motion(self, patient_motion: List[MotionState]) -> ComparisonResult:
        """
        Compare patient motion against reference.
        
        This method performs temporal alignment and calculates multiple metrics
        to provide comprehensive rehabilitation feedback.
        
        Args:
            patient_motion: List of MotionState objects from patient's performance
            
        Returns:
            ComparisonResult with assessment metrics
        """
        if not self.reference_loaded:
            logger.warning("No reference motion loaded for comparison")
            return ComparisonResult(
                similarity_score=0.0,
                angle_error=999.0,
                timing_ratio=0.0,
                smoothness_score=0.0,
                error_type="No reference motion available",
                recommendations=["Please record reference motion first"]
            )
        
        if len(patient_motion) < 5:
            return ComparisonResult(
                similarity_score=0.0,
                angle_error=999.0,
                timing_ratio=0.0,
                smoothness_score=0.0,
                error_type="Insufficient patient data",
                recommendations=["Perform complete repetition"]
            )
        
        # Step 1: Temporal alignment using Dynamic Time Warping (simplified)
        aligned_patient, aligned_reference = self._align_sequences(patient_motion, self.reference_motion)
        
        # Step 2: Calculate angle error (RMS)
        angle_error = self._calculate_angle_error(aligned_patient, aligned_reference)
        
        # Step 3: Calculate timing ratio
        timing_ratio = len(patient_motion) / len(self.reference_motion)
        
        # Step 4: Calculate smoothness score
        smoothness_score = self._calculate_smoothness(patient_motion)
        
        # Step 5: Determine overall similarity
        similarity_score = self._calculate_similarity(angle_error, timing_ratio, smoothness_score)
        
        # Step 6: Generate error type and recommendations
        error_type, recommendations = self._generate_feedback(
            angle_error, timing_ratio, smoothness_score, aligned_patient, aligned_reference
        )
        
        return ComparisonResult(
            similarity_score=similarity_score,
            angle_error=angle_error,
            timing_ratio=timing_ratio,
            smoothness_score=smoothness_score,
            error_type=error_type,
            recommendations=recommendations
        )
    
    def _align_sequences(self, patient: List[MotionState], reference: List[MotionState]) -> Tuple[List[float], List[float]]:
        """
        Align patient and reference sequences using simplified DTW.
        
        Allows comparison even when patient moves at different speed.
        
        Returns:
            Tuple of aligned angle sequences
        """
        # Extract angle sequences
        patient_angles = [s.elbow_angle for s in patient]
        reference_angles = [s.elbow_angle for s in reference]
        
        # Simple linear interpolation alignment
        # For production, consider full DTW implementation
        if len(patient_angles) != len(reference_angles):
            # Resample patient to match reference length
            patient_indices = np.linspace(0, len(patient_angles) - 1, len(reference_angles))
            patient_aligned = np.interp(patient_indices, range(len(patient_angles)), patient_angles)
            reference_aligned = reference_angles
        else:
            patient_aligned = patient_angles
            reference_aligned = reference_angles
        
        return patient_aligned, reference_aligned
    
    def _calculate_angle_error(self, patient_angles: List[float], reference_angles: List[float]) -> float:
        """
        Calculate RMS error between angle sequences.
        
        Args:
            patient_angles, reference_angles: Aligned angle sequences
            
        Returns:
            RMS error in degrees
        """
        errors = np.array(patient_angles) - np.array(reference_angles)
        rms_error = np.sqrt(np.mean(errors**2))
        return float(rms_error)
    
    def _calculate_smoothness(self, motion: List[MotionState]) -> float:
        """
        Calculate movement smoothness score.
        
        Smoother movements indicate better motor control.
        Uses jerk (derivative of acceleration) as smoothness metric.
        
        Args:
            motion: Patient motion sequence
            
        Returns:
            Smoothness score (0-100, higher is better)
        """
        if len(motion) < 3:
            return 50.0
        
        # Extract angular velocities
        velocities = [s.angular_velocity for s in motion]
        
        # Calculate jerk (change in velocity)
        jerks = np.diff(velocities)
        
        # Lower jerk = smoother motion
        mean_jerk = np.mean(np.abs(jerks))
        
        # Map to 0-100 scale (empirically tuned threshold)
        smoothness = max(0, 100 - mean_jerk * 2)
        
        return float(smoothness)
    
    def _calculate_similarity(self, angle_error: float, timing_ratio: float, smoothness: float) -> float:
        """
        Calculate overall similarity score.
        
        Combines multiple metrics into single 0-100 score.
        
        Args:
            angle_error: RMS angle error
            timing_ratio: Patient speed / Reference speed
            smoothness: Smoothness score
            
        Returns:
            Overall similarity (0-100)
        """
        # Angle component (0-40 points)
        angle_score = max(0, 40 - angle_error * 2)
        
        # Timing component (0-30 points)
        timing_error = abs(1.0 - timing_ratio)
        timing_score = max(0, 30 - timing_error * 100)
        
        # Smoothness component (0-30 points)
        smoothness_score = smoothness * 0.3
        
        total = angle_score + timing_score + smoothness_score
        
        return float(np.clip(total, 0, 100))
    
    def _generate_feedback(
        self,
        angle_error: float,
        timing_ratio: float,
        smoothness: float,
        patient_angles: List[float],
        reference_angles: List[float]
    ) -> Tuple[str, List[str]]:
        """
        Generate human-readable error messages and recommendations.
        
        Returns:
            (error_type, list of recommendations)
        """
        recommendations = []
        
        # Check angle range
        patient_range = max(patient_angles) - min(patient_angles)
        reference_range = max(reference_angles) - min(reference_angles)
        
        if patient_range < reference_range * 0.8:
            recommendations.append("Try to achieve fuller range of motion")
        
        # Check timing
        if timing_ratio > 1.3:
            error_type = "Movement too slow"
            recommendations.append("Try to move at a steady, moderate pace")
        elif timing_ratio < 0.7:
            error_type = "Movement too fast"
            recommendations.append("Slow down to maintain control")
        elif angle_error > self.angle_tolerance:
            error_type = "Incorrect movement pattern"
            recommendations.append("Focus on matching the reference angle trajectory")
        elif smoothness < 60:
            error_type = "Movement not smooth"
            recommendations.append("Try to move more smoothly without jerky motions")
        else:
            error_type = "Good technique"
            recommendations.append("Keep up the good work!")
        
        # Check max angle achieved
        max_patient = max(patient_angles)
        max_reference = max(reference_angles)
        
        if max_patient < max_reference - 20:
            recommendations.append("Try to lift higher to match the target")
        
        return error_type, recommendations


class RehabilitationBackend:
    """
    Main backend controller for the rehabilitation system.
    
    Orchestrates all components:
    - Serial reading
    - Motion processing
    - Reference comparison
    - Output to game interface
    
    This is the primary interface for the game/visualization layer.
    """
    
    def __init__(self, serial_port: str, baudrate: int = 115200):
        """
        Initialize the rehabilitation backend.
        
        Args:
            serial_port: Arduino serial port
            baudrate: Communication speed
        """
        # Initialize components
        self.serial_reader = IMUSerialReader(serial_port, baudrate)
        self.motion_processor = MotionProcessor(sampling_rate=100.0)
        self.comparator = ReferenceComparator()
        
        # Output queues for game interface (thread-safe)
        self.motion_output_queue = queue.Queue(maxsize=100)
        self.comparison_output_queue = queue.Queue(maxsize=10)
        
        # Recording state
        self.recording = False
        self.recording_buffer: List[MotionState] = []
        
        # Mode: 'doctor' or 'patient'
        self.mode = 'patient'
        
        # Processing thread
        self.processing_thread = None
        self.processing_active = False
        
        logger.info("Rehabilitation backend initialized")
    
    def start(self):
        """Start the backend (begins reading and processing)."""
        logger.info("Starting rehabilitation backend")
        
        # Start serial reader
        self.serial_reader.start()
        
        # Start processing thread
        self.processing_active = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        logger.info("Backend started successfully")
    
    def stop(self):
        """Stop the backend gracefully."""
        logger.info("Stopping rehabilitation backend")
        
        self.processing_active = False
        self.serial_reader.stop()
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        logger.info("Backend stopped")
    
    def _processing_loop(self):
        """
        Main processing loop (runs in separate thread).
        
        Continuously:
        1. Reads IMU data
        2. Processes motion
        3. Compares to reference (in patient mode)
        4. Outputs to game interface
        """
        comparison_counter = 0
        COMPARISON_INTERVAL = 10  # Compare every N frames
        
        while self.processing_active:
            # Get raw IMU reading
            raw_reading = self.serial_reader.get_reading(timeout=0.1)
            
            if raw_reading is None:
                continue
            
            # Process to motion state
            motion_state = self.motion_processor.process_reading(raw_reading)
            
            # If recording, add to buffer
            if self.recording:
                self.recording_buffer.append(motion_state)
            
            # Send to game interface
            try:
                self.motion_output_queue.put_nowait(motion_state)
            except queue.Full:
                # Discard oldest if queue full
                try:
                    self.motion_output_queue.get_nowait()
                    self.motion_output_queue.put_nowait(motion_state)
                except queue.Empty:
                    pass
            
            # In patient mode, periodically compare to reference
            if self.mode == 'patient' and self.comparator.reference_loaded:
                comparison_counter += 1
                
                if comparison_counter >= COMPARISON_INTERVAL and len(self.recording_buffer) > 10:
                    # Compare recent motion
                    recent_motion = self.recording_buffer[-50:]  # Last 50 frames
                    comparison = self.comparator.compare_motion(recent_motion)
                    
                    try:
                        self.comparison_output_queue.put_nowait(comparison)
                    except queue.Full:
                        try:
                            self.comparison_output_queue.get_nowait()
                            self.comparison_output_queue.put_nowait(comparison)
                        except queue.Empty:
                            pass
                    
                    comparison_counter = 0
    
    def set_mode(self, mode: str):
        """
        Set operating mode.
        
        Args:
            mode: 'doctor' or 'patient'
        """
        if mode not in ['doctor', 'patient']:
            raise ValueError("Mode must be 'doctor' or 'patient'")
        
        self.mode = mode
        logger.info(f"Mode set to: {mode}")
    
    def start_recording(self):
        """Start recording motion (for capturing reference or patient motion)."""
        self.recording_buffer.clear()
        self.recording = True
        logger.info("Recording started")
    
    def stop_recording(self) -> List[MotionState]:
        """
        Stop recording and return captured motion.
        
        Returns:
            List of recorded MotionState objects
        """
        self.recording = False
        recorded = self.recording_buffer.copy()
        logger.info(f"Recording stopped. Captured {len(recorded)} frames")
        return recorded
    
    def save_reference_recording(self, filepath: str = "reference_motion.json"):
        """
        Save current recording as reference motion.
        
        Args:
            filepath: Path to save file
        """
        if not self.recording_buffer:
            logger.warning("No recording to save")
            return
        
        self.comparator.load_reference(self.recording_buffer)
        self.comparator.save_reference(filepath)
        logger.info("Reference motion saved")
    
    def load_reference(self, filepath: str = "reference_motion.json"):
        """
        Load reference motion from file.
        
        Args:
            filepath: Path to reference file
        """
        self.comparator.load_reference_from_file(filepath)
        logger.info("Reference motion loaded")
    
    def get_motion_state(self, timeout: float = 0.01) -> Optional[MotionState]:
        """
        Get latest processed motion state for visualization.
        
        This is the primary interface for the game layer to receive
        real-time position updates.
        
        Args:
            timeout: Maximum wait time
            
        Returns:
            MotionState or None
        """
        try:
            return self.motion_output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_comparison_result(self, timeout: float = 0.01) -> Optional[ComparisonResult]:
        """
        Get latest comparison result.
        
        Args:
            timeout: Maximum wait time
            
        Returns:
            ComparisonResult or None
        """
        try:
            return self.comparison_output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def is_connected(self) -> bool:
        """Check if Arduino is connected."""
        return self.serial_reader.connected


# Example usage
if __name__ == "__main__":
    # This demonstrates how to use the backend
    
    print("=" * 60)
    print("IMU Rehabilitation Backend - Test Mode")
    print("=" * 60)
    
    # Initialize backend (adjust port for your system)
    backend = RehabilitationBackend(serial_port='/dev/ttyUSB0', baudrate=115200)
    
    try:
        # Start backend
        backend.start()
        
        print("\nWaiting for Arduino connection...")
        time.sleep(3)
        
        if not backend.is_connected():
            print("ERROR: Arduino not connected. Check port and connection.")
            exit(1)
        
        print("Connected successfully!\n")
        
        # Example 1: Record reference motion (doctor mode)
        print("DOCTOR MODE: Recording reference motion")
        print("Perform a bicep curl now... (recording for 5 seconds)")
        
        backend.set_mode('doctor')
        backend.start_recording()
        
        start_time = time.time()
        while time.time() - start_time < 5.0:
            state = backend.get_motion_state()
            if state:
                print(f"Angle: {state.elbow_angle:.1f}° | Phase: {state.movement_phase} | Pos: ({state.position_2d[0]:.2f}, {state.position_2d[1]:.2f})")
            time.sleep(0.1)
        
        backend.stop_recording()
        backend.save_reference_recording("reference_motion.json")
        print("\nReference motion saved!\n")
        
        # Example 2: Patient mode with real-time comparison
        print("PATIENT MODE: Perform bicep curl for comparison")
        print("Performing for 5 seconds...")
        
        backend.set_mode('patient')
        backend.load_reference("reference_motion.json")
        backend.start_recording()
        
        start_time = time.time()
        while time.time() - start_time < 5.0:
            # Get motion state
            state = backend.get_motion_state()
            if state:
                print(f"Angle: {state.elbow_angle:.1f}° | Velocity: {state.angular_velocity:.1f}°/s")
            
            # Get comparison results
            comparison = backend.get_comparison_result()
            if comparison:
                print(f"  → Similarity: {comparison.similarity_score:.1f}% | Error: {comparison.error_type}")
                print(f"     Recommendations: {', '.join(comparison.recommendations)}")
            
            time.sleep(0.1)
        
        recorded = backend.stop_recording()
        
        # Final comparison
        print("\nFinal Assessment:")
        final_comparison = backend.comparator.compare_motion(recorded)
        print(f"Similarity Score: {final_comparison.similarity_score:.1f}/100")
        print(f"Angle Error: {final_comparison.angle_error:.2f}°")
        print(f"Timing Ratio: {final_comparison.timing_ratio:.2f}x")
        print(f"Smoothness: {final_comparison.smoothness_score:.1f}/100")
        print(f"Assessment: {final_comparison.error_type}")
        print("Recommendations:")
        for rec in final_comparison.recommendations:
            print(f"  • {rec}")
    
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    
    finally:
        backend.stop()
        print("\nBackend stopped. Goodbye!")
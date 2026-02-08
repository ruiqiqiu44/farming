"""
Flask Backend Server for IMU Rehabilitation System
==================================================

This server bridges the Python IMU processor with the React/TypeScript frontend.
It provides:
- WebSocket real-time sensor data streaming
- REST API for exercise management
- Patient position tracking
- Exercise completion validation

Run with: python flask_backend.py
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import threading
import time
import json
from typing import Dict, Optional
import logging

# Import our IMU processor
from imu_processor import RehabilitationBackend, MotionState, ComparisonResult
from imu_simulator import patch_serial_for_simulation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'rehab-secret-key'
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global backend instance
backend: Optional[RehabilitationBackend] = None
backend_thread: Optional[threading.Thread] = None
is_running = False

# Exercise state
current_exercise = {
    'active': False,
    'patient_name': '',
    'exercise_name': '',
    'target_reps': 0,
    'completed_reps': 0,
    'water_earned': 0,
    'sun_earned': 0,
    'last_rep_time': 0
}

# Sensor positions (for the two circles on screen)
sensor_positions = {
    'left': {'x': 0.15, 'y': 0.55, 'active': False},
    'right': {'x': 0.85, 'y': 0.55, 'active': False}
}

# Target zones on avatar (where sensors should align)
target_zones = {
    'bicep_curl': {
        'left': {'x': 0.15, 'y': 0.45, 'radius': 0.08},
        'right': {'x': 0.85, 'y': 0.45, 'radius': 0.08}
    },
    'shoulder_press': {
        'left': {'x': 0.15, 'y': 0.25, 'radius': 0.08},
        'right': {'x': 0.85, 'y': 0.25, 'radius': 0.08}
    }
}


def initialize_backend(use_simulation: bool = True, serial_port: str = '/dev/ttyUSB0'):
    """Initialize the IMU backend."""
    global backend
    
    try:
        if use_simulation:
            logger.info("Starting in SIMULATION mode")
            patch_serial_for_simulation('perfect_technique')
            serial_port = '/dev/null'  # Dummy port for simulation
        
        backend = RehabilitationBackend(serial_port=serial_port)
        backend.start()
        
        logger.info("Backend initialized successfully")
        return True
    
    except Exception as e:
        logger.error(f"Failed to initialize backend: {e}")
        return False


def check_alignment(sensor_pos: Dict, target_pos: Dict) -> bool:
    """Check if sensor position is aligned with target zone."""
    distance = ((sensor_pos['x'] - target_pos['x'])**2 + 
                (sensor_pos['y'] - target_pos['y'])**2)**0.5
    return distance < target_pos['radius']


def process_motion_data():
    """Background thread to process IMU data and emit to frontend."""
    global is_running, current_exercise, sensor_positions
    
    logger.info("Motion processing thread started")
    last_rep_angle = 0
    angle_threshold_low = 30  # Extended position
    angle_threshold_high = 130  # Flexed position
    in_rep = False
    
    while is_running:
        if not backend:
            time.sleep(0.1)
            continue
        
        try:
            # Get motion state from IMU
            motion = backend.get_motion_state(timeout=0.01)
            
            if motion and current_exercise['active']:
                # Map elbow angle to screen position
                # For bicep curl: angle 0-180 maps to y position
                angle_normalized = motion.elbow_angle / 180.0
                
                # Update sensor positions based on motion
                # Left and right sensors move symmetrically for bicep curl
                sensor_positions['left']['y'] = 0.3 + (angle_normalized * 0.4)
                sensor_positions['right']['y'] = 0.3 + (angle_normalized * 0.4)
                
                # Check if in target zone
                exercise_type = current_exercise['exercise_name'].lower()
                if 'curl' in exercise_type:
                    targets = target_zones['bicep_curl']
                else:
                    targets = target_zones['bicep_curl']  # Default
                
                left_aligned = check_alignment(sensor_positions['left'], targets['left'])
                right_aligned = check_alignment(sensor_positions['right'], targets['right'])
                
                sensor_positions['left']['active'] = left_aligned
                sensor_positions['right']['active'] = right_aligned
                
                # Detect rep completion
                # Rep = go from extended to flexed and back
                if motion.elbow_angle < angle_threshold_low and not in_rep:
                    # Started extension phase
                    in_rep = True
                
                elif motion.elbow_angle > angle_threshold_high and in_rep:
                    # Reached flexion, check if aligned
                    current_time = time.time()
                    
                    # Only count if both sensors were aligned at top of movement
                    if left_aligned and right_aligned:
                        if current_time - current_exercise['last_rep_time'] > 1.0:  # Debounce
                            current_exercise['completed_reps'] += 1
                            current_exercise['water_earned'] += 10
                            current_exercise['sun_earned'] += 5
                            current_exercise['last_rep_time'] = current_time
                            
                            logger.info(f"Rep completed! Total: {current_exercise['completed_reps']}")
                            
                            # Emit rep completion event
                            socketio.emit('rep_completed', {
                                'completed_reps': current_exercise['completed_reps'],
                                'water_earned': current_exercise['water_earned'],
                                'sun_earned': current_exercise['sun_earned']
                            })
                    
                    in_rep = False
                
                # Emit sensor data to frontend
                socketio.emit('sensor_update', {
                    'sensors': sensor_positions,
                    'angle': motion.elbow_angle,
                    'velocity': motion.angular_velocity,
                    'phase': motion.movement_phase,
                    'both_aligned': left_aligned and right_aligned
                })
            
            # Get comparison results if in patient mode
            if current_exercise['active']:
                comparison = backend.get_comparison_result(timeout=0.01)
                if comparison:
                    socketio.emit('comparison_update', {
                        'similarity_score': comparison.similarity_score,
                        'angle_error': comparison.angle_error,
                        'timing_ratio': comparison.timing_ratio,
                        'smoothness_score': comparison.smoothness_score,
                        'feedback': comparison.error_type,
                        'recommendations': comparison.recommendations
                    })
        
        except Exception as e:
            logger.error(f"Error in motion processing: {e}")
        
        time.sleep(0.016)  # ~60 FPS


# ========== REST API ENDPOINTS ==========

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get backend status."""
    return jsonify({
        'running': is_running,
        'backend_connected': backend is not None and backend.is_connected() if backend else False,
        'exercise_active': current_exercise['active']
    })


@app.route('/api/start', methods=['POST'])
def start_backend():
    """Start the IMU backend."""
    global is_running, backend_thread
    
    data = request.json or {}
    use_simulation = data.get('simulation', True)
    serial_port = data.get('serial_port', '/dev/ttyUSB0')
    
    if initialize_backend(use_simulation, serial_port):
        is_running = True
        backend_thread = threading.Thread(target=process_motion_data, daemon=True)
        backend_thread.start()
        
        return jsonify({'success': True, 'message': 'Backend started'})
    else:
        return jsonify({'success': False, 'message': 'Failed to start backend'}), 500


@app.route('/api/stop', methods=['POST'])
def stop_backend():
    """Stop the IMU backend."""
    global is_running, backend
    
    is_running = False
    if backend:
        backend.stop()
        backend = None
    
    return jsonify({'success': True, 'message': 'Backend stopped'})


@app.route('/api/exercise/start', methods=['POST'])
def start_exercise():
    """Start an exercise session."""
    data = request.json
    
    if not backend:
        return jsonify({'success': False, 'message': 'Backend not initialized'}), 400
    
    # Set mode to patient
    backend.set_mode('patient')
    
    # Try to load reference motion
    exercise_name = data.get('exercise_name', 'bicep_curl')
    reference_file = f"references/{exercise_name}_reference.json"
    
    try:
        backend.load_reference(reference_file)
    except:
        logger.warning(f"No reference file found for {exercise_name}, using default")
    
    # Start recording patient motion
    backend.start_recording()
    
    # Update exercise state
    current_exercise.update({
        'active': True,
        'patient_name': data.get('patient_name', 'Unknown'),
        'exercise_name': data.get('exercise_name', 'Bicep Curls'),
        'target_reps': data.get('target_reps', 10),
        'completed_reps': 0,
        'water_earned': 0,
        'sun_earned': 0,
        'last_rep_time': 0
    })
    
    logger.info(f"Exercise started: {current_exercise}")
    
    return jsonify({
        'success': True,
        'message': 'Exercise started',
        'exercise': current_exercise
    })


@app.route('/api/exercise/stop', methods=['POST'])
def stop_exercise():
    """Stop current exercise session."""
    if not backend:
        return jsonify({'success': False, 'message': 'Backend not initialized'}), 400
    
    # Stop recording
    recorded_motion = backend.stop_recording()
    
    # Get final assessment
    final_assessment = None
    if len(recorded_motion) > 10:
        final_assessment = backend.comparator.compare_motion(recorded_motion)
    
    # Mark exercise as inactive
    current_exercise['active'] = False
    
    result = {
        'success': True,
        'completed_reps': current_exercise['completed_reps'],
        'target_reps': current_exercise['target_reps'],
        'water_earned': current_exercise['water_earned'],
        'sun_earned': current_exercise['sun_earned'],
        'frames_recorded': len(recorded_motion)
    }
    
    if final_assessment:
        result['assessment'] = {
            'similarity_score': final_assessment.similarity_score,
            'angle_error': final_assessment.angle_error,
            'timing_ratio': final_assessment.timing_ratio,
            'smoothness_score': final_assessment.smoothness_score,
            'feedback': final_assessment.error_type,
            'recommendations': final_assessment.recommendations
        }
    
    logger.info(f"Exercise stopped: {result}")
    
    return jsonify(result)


@app.route('/api/exercise/status', methods=['GET'])
def get_exercise_status():
    """Get current exercise status."""
    return jsonify(current_exercise)


@app.route('/api/reference/record', methods=['POST'])
def record_reference():
    """Record a reference motion (doctor mode)."""
    data = request.json
    
    if not backend:
        return jsonify({'success': False, 'message': 'Backend not initialized'}), 400
    
    exercise_name = data.get('exercise_name', 'bicep_curl')
    duration = data.get('duration', 5)  # seconds
    
    # Set to doctor mode
    backend.set_mode('doctor')
    backend.start_recording()
    
    logger.info(f"Recording reference for {exercise_name} ({duration}s)...")
    
    return jsonify({
        'success': True,
        'message': f'Recording reference for {duration} seconds',
        'exercise_name': exercise_name
    })


@app.route('/api/reference/save', methods=['POST'])
def save_reference():
    """Save recorded reference motion."""
    data = request.json
    
    if not backend:
        return jsonify({'success': False, 'message': 'Backend not initialized'}), 400
    
    exercise_name = data.get('exercise_name', 'bicep_curl')
    
    # Stop recording and save
    backend.stop_recording()
    reference_file = f"references/{exercise_name}_reference.json"
    
    try:
        import os
        os.makedirs('references', exist_ok=True)
        backend.save_reference_recording(reference_file)
        
        logger.info(f"Reference saved to {reference_file}")
        
        return jsonify({
            'success': True,
            'message': 'Reference motion saved',
            'filename': reference_file
        })
    except Exception as e:
        logger.error(f"Failed to save reference: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


# ========== WEBSOCKET EVENTS ==========

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info("Client connected")
    emit('connection_response', {'status': 'connected'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info("Client disconnected")


@socketio.on('request_sensor_data')
def handle_sensor_request():
    """Client requesting current sensor data."""
    emit('sensor_update', {
        'sensors': sensor_positions,
        'exercise_active': current_exercise['active']
    })


if __name__ == '__main__':
    import os
    
    print("="*70)
    print("  IMU REHABILITATION BACKEND SERVER")
    print("="*70)
    print("\n  Starting Flask + SocketIO server...")
    print(f"  Backend API: http://localhost:5000")
    print(f"  WebSocket: ws://localhost:5000")
    print("\n  Endpoints:")
    print("    GET  /api/status")
    print("    POST /api/start")
    print("    POST /api/stop")
    print("    POST /api/exercise/start")
    print("    POST /api/exercise/stop")
    print("    GET  /api/exercise/status")
    print("    POST /api/reference/record")
    print("    POST /api/reference/save")
    print("\n  WebSocket Events:")
    print("    → sensor_update (real-time sensor positions)")
    print("    → rep_completed (when rep is detected)")
    print("    → comparison_update (exercise assessment)")
    print("="*70 + "\n")
    
    # Auto-start in simulation mode
    initialize_backend(use_simulation=True)
    is_running = True
    backend_thread = threading.Thread(target=process_motion_data, daemon=True)
    backend_thread.start()
    
    # Run server
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
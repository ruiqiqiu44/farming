#!/usr/bin/env python3
"""
Rehabilitation System Demo (Simulation Mode)
============================================

This script demonstrates the rehabilitation system using simulated IMU data.
No Arduino hardware required - perfect for:
- Development
- Testing
- Demonstrations
- Algorithm validation

Run this to see the complete system in action!
"""

import sys
import time

# Enable simulation mode BEFORE importing the backend
from imu_simulator import patch_serial_for_simulation

# Choose scenario (see imu_simulator.py for all options)
SIMULATION_SCENARIO = 'perfect_technique'  # Change to test different cases

# Patch serial module to use simulation
patch_serial_for_simulation(SIMULATION_SCENARIO)

# Now import the backend (will use simulated serial)
from imu_processor import RehabilitationBackend, MotionState, ComparisonResult


def demo_backend_only():
    """
    Demo using backend directly without game interface.
    
    Shows how to integrate the backend into custom applications.
    """
    print("\n" + "="*60)
    print("DEMO 1: Backend Only (No GUI)")
    print("="*60 + "\n")
    
    # Initialize backend with fake port (will be simulated)
    backend = RehabilitationBackend(serial_port='/dev/null', baudrate=115200)
    
    try:
        # Start backend
        backend.start()
        print("Backend started (simulated data)")
        time.sleep(1)
        
        # --- DOCTOR MODE: Record Reference ---
        print("\n[Doctor Mode] Recording reference motion (5 seconds)...")
        backend.set_mode('doctor')
        backend.start_recording()
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < 5.0:
            motion = backend.get_motion_state(timeout=0.1)
            if motion:
                frame_count += 1
                if frame_count % 10 == 0:  # Print every 10th frame
                    print(f"  Frame {frame_count}: Angle={motion.elbow_angle:.1f}°, "
                          f"Phase={motion.movement_phase}, "
                          f"Velocity={motion.angular_velocity:.1f}°/s")
        
        backend.stop_recording()
        backend.save_reference_recording("demo_reference.json")
        print(f"✓ Recorded {frame_count} frames")
        
        # --- PATIENT MODE: Compare Motion ---
        print("\n[Patient Mode] Comparing to reference (5 seconds)...")
        backend.set_mode('patient')
        backend.load_reference("demo_reference.json")
        backend.start_recording()
        
        start_time = time.time()
        comparison_count = 0
        
        while time.time() - start_time < 5.0:
            # Get motion
            motion = backend.get_motion_state(timeout=0.01)
            
            # Get comparison results
            comparison = backend.get_comparison_result(timeout=0.01)
            if comparison:
                comparison_count += 1
                print(f"\n  Comparison #{comparison_count}:")
                print(f"    Score: {comparison.similarity_score:.1f}%")
                print(f"    Angle Error: {comparison.angle_error:.2f}°")
                print(f"    Timing: {comparison.timing_ratio:.2f}x")
                print(f"    Smoothness: {comparison.smoothness_score:.1f}%")
                print(f"    Assessment: {comparison.error_type}")
                if comparison.recommendations:
                    print(f"    Tip: {comparison.recommendations[0]}")
        
        recorded = backend.stop_recording()
        
        # Final assessment
        print("\n" + "="*60)
        print("FINAL ASSESSMENT")
        print("="*60)
        
        final = backend.comparator.compare_motion(recorded)
        print(f"Overall Score: {final.similarity_score:.1f}/100")
        print(f"Angle Error: {final.angle_error:.2f}°")
        print(f"Timing Ratio: {final.timing_ratio:.2f}x")
        print(f"Smoothness: {final.smoothness_score:.1f}/100")
        print(f"\nAssessment: {final.error_type}")
        print("Recommendations:")
        for i, rec in enumerate(final.recommendations, 1):
            print(f"  {i}. {rec}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        backend.stop()
        print("\n✓ Backend demo complete")


def demo_data_extraction():
    """
    Demo showing how to extract and process motion data.
    
    Useful for:
    - Research
    - Custom analytics
    - Export to other systems
    """
    print("\n" + "="*60)
    print("DEMO 2: Data Extraction & Analysis")
    print("="*60 + "\n")
    
    backend = RehabilitationBackend(serial_port='/dev/null')
    
    try:
        backend.start()
        time.sleep(1)
        
        # Collect data
        print("Collecting motion data (3 seconds)...")
        backend.start_recording()
        
        motion_data = []
        start_time = time.time()
        
        while time.time() - start_time < 3.0:
            motion = backend.get_motion_state(timeout=0.1)
            if motion:
                motion_data.append(motion)
        
        backend.stop_recording()
        
        # Analyze collected data
        print(f"\n✓ Collected {len(motion_data)} motion samples")
        
        if motion_data:
            # Extract angle trajectory
            angles = [m.elbow_angle for m in motion_data]
            velocities = [m.angular_velocity for m in motion_data]
            
            print("\nMotion Statistics:")
            print(f"  Min Angle: {min(angles):.1f}°")
            print(f"  Max Angle: {max(angles):.1f}°")
            print(f"  Range: {max(angles) - min(angles):.1f}°")
            print(f"  Avg Angle: {sum(angles)/len(angles):.1f}°")
            print(f"  Max Velocity: {max(abs(v) for v in velocities):.1f}°/s")
            
            # Count movement phases
            phases = [m.movement_phase for m in motion_data]
            flexion_count = phases.count('flexion')
            extension_count = phases.count('extension')
            hold_count = phases.count('hold')
            
            print("\nMovement Phases:")
            print(f"  Flexion: {flexion_count} frames ({100*flexion_count/len(phases):.1f}%)")
            print(f"  Extension: {extension_count} frames ({100*extension_count/len(phases):.1f}%)")
            print(f"  Hold: {hold_count} frames ({100*hold_count/len(phases):.1f}%)")
            
            # Export sample data
            print("\nSample trajectory (first 10 frames):")
            print("  Frame | Timestamp | Angle  | Velocity | Phase")
            print("  " + "-"*50)
            for i, m in enumerate(motion_data[:10], 1):
                print(f"  {i:5d} | {m.timestamp:9.0f} | {m.elbow_angle:6.1f}° | "
                      f"{m.angular_velocity:8.1f}°/s | {m.movement_phase:9s}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted")
    
    finally:
        backend.stop()
        print("\n✓ Data extraction demo complete")


def demo_realtime_console():
    """
    Real-time console display of motion data.
    
    Shows live updating of sensor readings in terminal.
    """
    print("\n" + "="*60)
    print("DEMO 3: Real-Time Console Display")
    print("="*60 + "\n")
    print("Press Ctrl+C to stop\n")
    
    backend = RehabilitationBackend(serial_port='/dev/null')
    
    try:
        backend.start()
        time.sleep(1)
        
        print("Angle | Velocity | Phase     | Position (2D)     | Accel")
        print("-" * 65)
        
        while True:
            motion = backend.get_motion_state(timeout=0.1)
            if motion:
                # Create visual angle bar
                angle_pct = motion.elbow_angle / 180.0
                bar_length = int(angle_pct * 20)
                bar = "█" * bar_length + "░" * (20 - bar_length)
                
                # Format output
                print(f"\r{motion.elbow_angle:5.1f}° |"
                      f"{motion.angular_velocity:8.1f}°/s |"
                      f"{motion.movement_phase:9s} |"
                      f"({motion.position_2d[0]:5.2f}, {motion.position_2d[1]:5.2f}) |"
                      f"{motion.acceleration_magnitude:5.2f} m/s² "
                      f"{bar}", end="", flush=True)
            
            time.sleep(0.05)
    
    except KeyboardInterrupt:
        print("\n\n")
    
    finally:
        backend.stop()
        print("✓ Real-time demo complete")


def demo_comparison_live():
    """
    Live comparison with visual feedback in console.
    """
    print("\n" + "="*60)
    print("DEMO 4: Live Comparison Feedback")
    print("="*60 + "\n")
    
    backend = RehabilitationBackend(serial_port='/dev/null')
    
    try:
        backend.start()
        time.sleep(1)
        
        # Record reference
        print("Recording reference (3 seconds)...")
        backend.set_mode('doctor')
        backend.start_recording()
        time.sleep(3)
        backend.stop_recording()
        backend.save_reference_recording("demo_reference.json")
        print("✓ Reference saved\n")
        
        # Start patient mode
        print("Patient mode - comparing in real-time...")
        print("Press Ctrl+C to stop\n")
        backend.set_mode('patient')
        backend.load_reference("demo_reference.json")
        backend.start_recording()
        
        print("Score | Error  | Timing | Smooth | Assessment")
        print("-" * 60)
        
        last_score = None
        
        while True:
            comparison = backend.get_comparison_result(timeout=0.1)
            if comparison:
                # Visual score bar
                score_pct = comparison.similarity_score / 100.0
                bar_length = int(score_pct * 15)
                
                if score_pct >= 0.8:
                    bar_char = "█"  # Excellent
                elif score_pct >= 0.6:
                    bar_char = "▓"  # Good
                else:
                    bar_char = "░"  # Needs improvement
                
                score_bar = bar_char * bar_length + "░" * (15 - bar_length)
                
                # Show trend
                if last_score:
                    if comparison.similarity_score > last_score + 5:
                        trend = "↗"
                    elif comparison.similarity_score < last_score - 5:
                        trend = "↘"
                    else:
                        trend = "→"
                else:
                    trend = " "
                
                last_score = comparison.similarity_score
                
                print(f"\r{comparison.similarity_score:5.1f} {trend} |"
                      f"{comparison.angle_error:6.1f}° |"
                      f"{comparison.timing_ratio:6.2f}x |"
                      f"{comparison.smoothness_score:6.1f} |"
                      f"{comparison.error_type:20s} "
                      f"{score_bar}", end="", flush=True)
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\n")
    
    finally:
        backend.stop_recording()
        backend.stop()
        print("✓ Live comparison demo complete")


def main():
    """Main demo menu."""
    print("\n" + "="*60)
    print("IMU REHABILITATION SYSTEM - SIMULATION DEMO")
    print("="*60)
    print(f"\nSimulation Scenario: {SIMULATION_SCENARIO}")
    print("\nThis demo runs without Arduino hardware.")
    print("All sensor data is simulated for testing.")
    print("="*60)
    
    demos = {
        '1': ('Backend Integration', demo_backend_only),
        '2': ('Data Extraction & Analysis', demo_data_extraction),
        '3': ('Real-Time Console Display', demo_realtime_console),
        '4': ('Live Comparison Feedback', demo_comparison_live),
    }
    
    print("\nAvailable Demos:")
    for key, (name, _) in demos.items():
        print(f"  {key}. {name}")
    print("  5. Run All Demos")
    print("  Q. Quit")
    
    while True:
        choice = input("\nSelect demo (1-5, Q to quit): ").strip().upper()
        
        if choice == 'Q':
            print("Goodbye!")
            break
        elif choice in demos:
            _, demo_func = demos[choice]
            demo_func()
            
            input("\nPress Enter to continue...")
        elif choice == '5':
            # Run all demos
            for _, demo_func in demos.values():
                demo_func()
                print("\n" + "="*60 + "\n")
                time.sleep(2)
            
            print("\n✓ All demos complete!")
            break
        else:
            print("Invalid choice. Please select 1-5 or Q.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting demo...")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
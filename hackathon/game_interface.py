"""
Rehabilitation Game Interface - Visual Feedback System
======================================================

This module provides a game-like interface for the rehabilitation system,
displaying real-time sensor positions, reference animations, and feedback.

Uses Pygame for rendering but can be adapted to Unity or other engines.
"""

import pygame
import sys
import time
from typing import Optional
import math

from imu_processor import RehabilitationBackend, MotionState, ComparisonResult


class RehabGameInterface:
    """
    Game interface for rehabilitation visualization.
    
    Features:
    - Real-time patient sensor visualization (moving circles)
    - Doctor reference animation playback
    - Visual feedback on movement quality
    - Score display
    """
    
    def __init__(self, backend: RehabilitationBackend, width: int = 1200, height: int = 800):
        """
        Initialize the game interface.
        
        Args:
            backend: RehabilitationBackend instance
            width, height: Screen dimensions
        """
        self.backend = backend
        self.width = width
        self.height = height
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Rehabilitation Training System")
        self.clock = pygame.time.Clock()
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 150, 255)
        self.YELLOW = (255, 255, 0)
        self.GRAY = (128, 128, 128)
        self.LIGHT_GRAY = (200, 200, 200)
        
        # Fonts
        self.font_large = pygame.font.Font(None, 72)
        self.font_medium = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # Visualization parameters
        self.center_x = width // 2
        self.center_y = height // 2
        self.scale = 300  # Pixels per normalized unit
        
        # State
        self.current_motion_state: Optional[MotionState] = None
        self.current_comparison: Optional[ComparisonResult] = None
        self.reference_animation_frame = 0
        self.reference_animation_speed = 1.0
        
        # Reference motion for animation
        self.reference_motion = []
        
        # Performance tracking
        self.score_history = []
        
        print("Game interface initialized")
    
    def load_reference_animation(self, filepath: str = "reference_motion.json"):
        """
        Load reference motion for animation playback.
        
        Args:
            filepath: Path to reference motion JSON
        """
        self.backend.load_reference(filepath)
        
        # Store reference for animation
        if self.backend.comparator.reference_loaded:
            self.reference_motion = self.backend.comparator.reference_motion
            print(f"Loaded reference animation with {len(self.reference_motion)} frames")
    
    def run_doctor_mode(self, recording_duration: float = 10.0):
        """
        Run doctor mode: record reference motion.
        
        Args:
            recording_duration: How long to record (seconds)
        """
        self.backend.set_mode('doctor')
        
        print("\n" + "="*60)
        print("DOCTOR MODE - Recording Reference Motion")
        print("="*60)
        print(f"Recording for {recording_duration} seconds...")
        print("Perform the bicep curl exercise correctly.")
        print("="*60 + "\n")
        
        # Countdown
        for i in range(3, 0, -1):
            self.screen.fill(self.BLACK)
            countdown_text = self.font_large.render(f"Starting in {i}...", True, self.YELLOW)
            self.screen.blit(countdown_text, (self.width // 2 - 200, self.height // 2))
            pygame.display.flip()
            time.sleep(1)
        
        # Start recording
        self.backend.start_recording()
        start_time = time.time()
        
        running = True
        while running and (time.time() - start_time) < recording_duration:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Get current motion state
            motion_state = self.backend.get_motion_state()
            if motion_state:
                self.current_motion_state = motion_state
            
            # Render
            self._render_doctor_mode(recording_duration - (time.time() - start_time))
            
            # Update display
            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS
        
        # Stop recording and save
        self.backend.stop_recording()
        self.backend.save_reference_recording("reference_motion.json")
        
        # Show completion message
        self.screen.fill(self.BLACK)
        complete_text = self.font_large.render("Reference Recorded!", True, self.GREEN)
        self.screen.blit(complete_text, (self.width // 2 - 300, self.height // 2))
        pygame.display.flip()
        time.sleep(2)
        
        print("\nReference motion saved successfully!")
    
    def run_patient_mode(self):
        """
        Run patient mode: compare performance to reference.
        """
        self.backend.set_mode('patient')
        
        # Load reference
        self.load_reference_animation("reference_motion.json")
        
        print("\n" + "="*60)
        print("PATIENT MODE - Training with Visual Feedback")
        print("="*60)
        print("Match your movement to the reference animation.")
        print("Press SPACE to start/stop exercise")
        print("Press ESC to exit")
        print("="*60 + "\n")
        
        running = True
        exercising = False
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        if not exercising:
                            # Start exercise
                            exercising = True
                            self.backend.start_recording()
                            self.score_history.clear()
                            print("Exercise started!")
                        else:
                            # Stop exercise
                            exercising = False
                            recorded = self.backend.stop_recording()
                            
                            # Final assessment
                            if len(recorded) > 10:
                                final = self.backend.comparator.compare_motion(recorded)
                                print(f"\nFinal Score: {final.similarity_score:.1f}/100")
                                print(f"Assessment: {final.error_type}")
                                for rec in final.recommendations:
                                    print(f"  • {rec}")
                            
                            print("Exercise stopped!")
            
            # Get current motion state
            motion_state = self.backend.get_motion_state()
            if motion_state:
                self.current_motion_state = motion_state
            
            # Get comparison results
            comparison = self.backend.get_comparison_result()
            if comparison:
                self.current_comparison = comparison
                self.score_history.append(comparison.similarity_score)
                
                # Keep only recent scores
                if len(self.score_history) > 20:
                    self.score_history.pop(0)
            
            # Update reference animation
            if len(self.reference_motion) > 0:
                self.reference_animation_frame += self.reference_animation_speed
                if self.reference_animation_frame >= len(self.reference_motion):
                    self.reference_animation_frame = 0
            
            # Render
            self._render_patient_mode(exercising)
            
            # Update display
            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS
    
    def _render_doctor_mode(self, time_remaining: float):
        """
        Render doctor mode interface.
        
        Args:
            time_remaining: Seconds left in recording
        """
        self.screen.fill(self.BLACK)
        
        # Title
        title = self.font_large.render("DOCTOR MODE - Recording", True, self.YELLOW)
        self.screen.blit(title, (50, 30))
        
        # Timer
        timer_text = self.font_medium.render(f"Time: {time_remaining:.1f}s", True, self.WHITE)
        self.screen.blit(timer_text, (50, 120))
        
        # Draw current sensor position
        if self.current_motion_state:
            self._draw_sensor_position(
                self.current_motion_state,
                color=self.GREEN,
                label="Current Position"
            )
            
            # Display metrics
            self._draw_motion_metrics(self.current_motion_state, x=50, y=200)
    
    def _render_patient_mode(self, exercising: bool):
        """
        Render patient mode interface.
        
        Args:
            exercising: Whether patient is currently performing exercise
        """
        self.screen.fill(self.BLACK)
        
        # Title
        title = self.font_large.render("PATIENT MODE - Training", True, self.BLUE)
        self.screen.blit(title, (50, 30))
        
        # Instructions
        if not exercising:
            instruction = self.font_small.render("Press SPACE to start exercise", True, self.WHITE)
            self.screen.blit(instruction, (50, 120))
        else:
            instruction = self.font_small.render("Exercising... Press SPACE to stop", True, self.GREEN)
            self.screen.blit(instruction, (50, 120))
        
        # Draw reference animation (doctor's movement)
        if len(self.reference_motion) > 0:
            ref_frame = int(self.reference_animation_frame) % len(self.reference_motion)
            ref_state = self.reference_motion[ref_frame]
            
            self._draw_sensor_position(
                ref_state,
                color=self.YELLOW,
                label="Reference (Doctor)",
                offset_x=-150
            )
        
        # Draw patient's current position
        if self.current_motion_state:
            self._draw_sensor_position(
                self.current_motion_state,
                color=self.GREEN if exercising else self.GRAY,
                label="Your Position",
                offset_x=150
            )
        
        # Draw comparison results and score
        if self.current_comparison and exercising:
            self._draw_comparison_panel(self.current_comparison)
        
        # Draw score graph
        if len(self.score_history) > 0 and exercising:
            self._draw_score_graph(self.score_history)
    
    def _draw_sensor_position(
        self,
        motion_state: MotionState,
        color: tuple,
        label: str,
        offset_x: int = 0
    ):
        """
        Draw sensor position as a visual indicator.
        
        This represents the patient's arm position in 2D space.
        
        Args:
            motion_state: Current motion state
            color: Circle color
            label: Text label
            offset_x: Horizontal offset for side-by-side comparison
        """
        # Calculate screen position from normalized coordinates
        pos_x = int(self.center_x + motion_state.position_2d[0] * self.scale + offset_x)
        pos_y = int(self.center_y + motion_state.position_2d[1] * self.scale)
        
        # Draw shoulder (fixed point)
        shoulder_x = self.center_x + offset_x
        shoulder_y = self.center_y + 50
        pygame.draw.circle(self.screen, self.GRAY, (shoulder_x, shoulder_y), 15)
        
        # Draw arm line
        pygame.draw.line(self.screen, color, (shoulder_x, shoulder_y), (pos_x, pos_y), 8)
        
        # Draw sensor circle (represents wrist/forearm)
        pygame.draw.circle(self.screen, color, (pos_x, pos_y), 25)
        pygame.draw.circle(self.screen, self.WHITE, (pos_x, pos_y), 25, 3)
        
        # Draw angle arc
        angle_rad = math.radians(motion_state.elbow_angle)
        arc_radius = 60
        arc_rect = pygame.Rect(
            shoulder_x - arc_radius,
            shoulder_y - arc_radius,
            arc_radius * 2,
            arc_radius * 2
        )
        
        # Draw arc showing angle
        pygame.draw.arc(
            self.screen,
            color,
            arc_rect,
            -math.pi / 2,  # Start angle (pointing down)
            -math.pi / 2 + angle_rad,  # End angle
            5
        )
        
        # Label
        label_surf = self.font_small.render(label, True, color)
        self.screen.blit(label_surf, (pos_x - 80, pos_y - 60))
        
        # Angle value
        angle_text = self.font_small.render(f"{motion_state.elbow_angle:.1f}°", True, self.WHITE)
        self.screen.blit(angle_text, (shoulder_x - 40, shoulder_y + 70))
    
    def _draw_motion_metrics(self, motion_state: MotionState, x: int, y: int):
        """
        Draw current motion metrics.
        
        Args:
            motion_state: Current motion state
            x, y: Position to draw
        """
        metrics = [
            f"Elbow Angle: {motion_state.elbow_angle:.1f}°",
            f"Angular Velocity: {motion_state.angular_velocity:.1f}°/s",
            f"Movement Phase: {motion_state.movement_phase}",
            f"Acceleration: {motion_state.acceleration_magnitude:.2f} m/s²"
        ]
        
        for i, metric in enumerate(metrics):
            text = self.font_small.render(metric, True, self.WHITE)
            self.screen.blit(text, (x, y + i * 40))
    
    def _draw_comparison_panel(self, comparison: ComparisonResult):
        """
        Draw comparison results panel.
        
        Args:
            comparison: Latest comparison result
        """
        panel_x = self.width - 400
        panel_y = 200
        panel_width = 350
        
        # Background
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, 400)
        pygame.draw.rect(self.screen, (30, 30, 30), panel_rect)
        pygame.draw.rect(self.screen, self.WHITE, panel_rect, 2)
        
        # Title
        title = self.font_medium.render("Performance", True, self.WHITE)
        self.screen.blit(title, (panel_x + 20, panel_y + 20))
        
        # Similarity score with color coding
        score = comparison.similarity_score
        if score >= 80:
            score_color = self.GREEN
        elif score >= 60:
            score_color = self.YELLOW
        else:
            score_color = self.RED
        
        score_text = self.font_large.render(f"{score:.0f}%", True, score_color)
        self.screen.blit(score_text, (panel_x + 120, panel_y + 80))
        
        # Error type
        error_text = self.font_small.render(comparison.error_type, True, self.WHITE)
        self.screen.blit(error_text, (panel_x + 20, panel_y + 170))
        
        # Metrics
        metrics = [
            f"Angle Error: {comparison.angle_error:.1f}°",
            f"Timing: {comparison.timing_ratio:.2f}x",
            f"Smoothness: {comparison.smoothness_score:.0f}%"
        ]
        
        y_offset = 210
        for metric in metrics:
            text = self.font_small.render(metric, True, self.LIGHT_GRAY)
            self.screen.blit(text, (panel_x + 20, panel_y + y_offset))
            y_offset += 35
        
        # Recommendations (show first one)
        if comparison.recommendations:
            rec_label = self.font_small.render("Tip:", True, self.YELLOW)
            self.screen.blit(rec_label, (panel_x + 20, panel_y + 330))
            
            # Wrap text if too long
            rec_text = comparison.recommendations[0]
            if len(rec_text) > 30:
                rec_text = rec_text[:27] + "..."
            
            rec = self.font_small.render(rec_text, True, self.WHITE)
            self.screen.blit(rec, (panel_x + 20, panel_y + 360))
    
    def _draw_score_graph(self, scores: list):
        """
        Draw real-time score history graph.
        
        Args:
            scores: List of recent similarity scores
        """
        graph_x = 50
        graph_y = self.height - 200
        graph_width = 400
        graph_height = 150
        
        # Background
        graph_rect = pygame.Rect(graph_x, graph_y, graph_width, graph_height)
        pygame.draw.rect(self.screen, (30, 30, 30), graph_rect)
        pygame.draw.rect(self.screen, self.WHITE, graph_rect, 2)
        
        # Title
        title = self.font_small.render("Score History", True, self.WHITE)
        self.screen.blit(title, (graph_x + 10, graph_y - 30))
        
        # Draw scores as line graph
        if len(scores) > 1:
            points = []
            for i, score in enumerate(scores):
                x = graph_x + int((i / len(scores)) * graph_width)
                y = graph_y + graph_height - int((score / 100) * graph_height)
                points.append((x, y))
            
            # Draw line
            pygame.draw.lines(self.screen, self.GREEN, False, points, 3)
            
            # Draw points
            for point in points:
                pygame.draw.circle(self.screen, self.GREEN, point, 4)
        
        # Reference line at 80%
        ref_y = graph_y + graph_height - int(0.8 * graph_height)
        pygame.draw.line(
            self.screen,
            self.YELLOW,
            (graph_x, ref_y),
            (graph_x + graph_width, ref_y),
            1
        )
    
    def shutdown(self):
        """Clean shutdown of game interface."""
        pygame.quit()
        print("Game interface closed")


def main():
    """
    Main entry point for the rehabilitation game interface.
    """
    print("=" * 60)
    print("IMU Rehabilitation Training System")
    print("=" * 60)
    
    # Configuration
    SERIAL_PORT = '/dev/ttyUSB0'  # Adjust for your system (COM3 on Windows)
    BAUDRATE = 115200
    
    # Initialize backend
    print("\nInitializing backend...")
    backend = RehabilitationBackend(serial_port=SERIAL_PORT, baudrate=BAUDRATE)
    
    try:
        # Start backend
        backend.start()
        
        print("Waiting for Arduino connection...")
        time.sleep(3)
        
        if not backend.is_connected():
            print("\nERROR: Could not connect to Arduino")
            print(f"Please check:")
            print(f"  - Arduino is connected to {SERIAL_PORT}")
            print(f"  - Baudrate matches ({BAUDRATE})")
            print(f"  - Serial monitor is closed")
            return
        
        print("✓ Arduino connected successfully!\n")
        
        # Initialize game interface
        game = RehabGameInterface(backend)
        
        # Ask user for mode
        print("Select mode:")
        print("  1. Doctor Mode (record reference)")
        print("  2. Patient Mode (training)")
        
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == '1':
            # Doctor mode
            duration = input("Recording duration in seconds (default 10): ").strip()
            if duration:
                duration = float(duration)
            else:
                duration = 10.0
            
            game.run_doctor_mode(recording_duration=duration)
            
            # Ask if they want to continue to patient mode
            cont = input("\nProceed to patient mode? (y/n): ").strip().lower()
            if cont == 'y':
                game.run_patient_mode()
        
        elif choice == '2':
            # Patient mode
            game.run_patient_mode()
        
        else:
            print("Invalid choice")
        
        # Cleanup
        game.shutdown()
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        backend.stop()
        print("\n✓ Backend stopped")
        print("Goodbye!")


if __name__ == "__main__":
    main()
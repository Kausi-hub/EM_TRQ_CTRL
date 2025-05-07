import time
import matplotlib.pyplot as plt

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, setpoint, measured_value):
        error = setpoint - measured_value
        self.integral += error
        derivative = error - self.prev_error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

class Motor:
    def __init__(self):
        self.position = 0
        self.speed = 0
        self.positions = []  # Store position data for plotting
        self.times = []      # Store time data for plotting

    def update_position(self, speed, time_step):
        self.position += speed * time_step  # Simulating motion
        self.positions.append(self.position)
        self.times.append(len(self.positions) * time_step)

def motor_position_control(target_position):
    position_pid = PIDController(kp=1.0, ki=0.1, kd=0.05)
    speed_pid = PIDController(kp=0.5, ki=0.05, kd=0.02)

    motor = Motor()
    time_step = 0.1  # Simulation time step
    max_iterations = 100

    for _ in range(max_iterations):
        position_error = target_position - motor.position

        if abs(position_error) < 0.5:  # Threshold for stopping
            print(f"Target position {target_position} reached. Final position: {motor.position:.2f}")
            break

        speed_command = position_pid.compute(target_position, motor.position)

        # Nested loop for speed control
        for _ in range(10):
            speed_adjustment = speed_pid.compute(speed_command, motor.speed)
            motor.speed += speed_adjustment
            motor.update_position(motor.speed, time_step)

    # Plot results
    plt.figure(figsize=(8, 4))
    plt.plot(motor.times, motor.positions, label="Motor Position")
    plt.axhline(y=target_position, color='r', linestyle='--', label="Target Position")
    plt.xlabel("Time (s)")
    plt.ylabel("Position")
    plt.title("Motor Position Control Simulation")
    plt.legend()
    plt.grid()
    plt.show()

motor_position_control(target_position=100)

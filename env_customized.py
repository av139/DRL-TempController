# env_customized.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import csv
import os
from mqtt_win import MQTTClient
from env_utils import EnvUtils
from skfuzzy import control as ctrl
import skfuzzy as fuzz

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0
        self.derivative = 0

    def compute(self, target_temp, actual_temp):
        # Implementation of the PID controller
        error = target_temp - actual_temp
        self.integral += error
        self.derivative = error - self.previous_error
        self.previous_error = error

        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * self.derivative)
        output = max(min(output, 100), -100)  # Clamp the PID output
        return output

class FuzzyController:
    def __init__(self):
        # Define fuzzy variables
        self.error = ctrl.Antecedent(np.arange(-30, 31, 1), 'error')
        self.output = ctrl.Consequent(np.arange(-100, 101, 1), 'output')

        # Define fuzzy sets
        self.error['negative'] = fuzz.trimf(self.error.universe, [-30, -5, 0])
        self.error['zero'] = fuzz.trimf(self.error.universe, [-5, 0, 5])
        self.error['positive'] = fuzz.trimf(self.error.universe, [0, 5, 30])

        self.output['cool'] = fuzz.trimf(self.output.universe, [-100, -100, 0])
        self.output['zero'] = fuzz.trimf(self.output.universe, [-100, 0, 100])
        self.output['heat'] = fuzz.trimf(self.output.universe, [0, 100, 100])

        # Define rules
        self.rule1 = ctrl.Rule(self.error['negative'], self.output['cool'])
        self.rule2 = ctrl.Rule(self.error['zero'], self.output['zero'])
        self.rule3 = ctrl.Rule(self.error['positive'], self.output['heat'])

        self.system = ctrl.ControlSystem([self.rule1, self.rule2, self.rule3])
        self.simulation = ctrl.ControlSystemSimulation(self.system)

    def compute(self, target_temp, actual_temp):
        error = target_temp - actual_temp
        self.simulation.input['error'] = error
        self.simulation.compute()
        output = self.simulation.output['output']
        return output

class CustomEnv(gym.Env):
    def __init__(self, broker_address, port, temp_topic, pwm_topic, target_temp_min, target_temp_max, algorithm, episode_max_steps=1000, render_mode='human', kp=2.0, ki=0.1, kd=0.01, temp_config=None, dist_config=None):
    
        super(CustomEnv, self).__init__()
        
        self.temp_config = temp_config
        self.render_mode = render_mode
        self.algorithm = algorithm
        
        self.dist_config = dist_config
        self.dist_dc = 0
        self.current_dist_steps = 0
        self.chance_of_zero = dist_config.get('chance_of_zero', 0) if dist_config else 0
        
        # Set action space and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([self.temp_config['target_temp_low'], self.temp_config['actual_temp_low'], self.temp_config['cooling_temp_low']], dtype=np.float32),
            high=np.array([self.temp_config['target_temp_high'], self.temp_config['actual_temp_high'], self.temp_config['cooling_temp_high']], dtype=np.float32)
        )

        # MQTT setup
        self.mqtt_client = MQTTClient(broker_address, port, temp_topic, pwm_topic)
        self.mqtt_client.connect()

        # Controller setup
        if self.algorithm == 'PID':
            self.controller = PIDController(kp, ki, kd)
        elif self.algorithm == 'Fuzzy':
            self.controller = FuzzyController()
        else:
            self.controller = None

        self.target_temp_min = target_temp_min
        self.target_temp_max = target_temp_max
        self.episode_max_steps = episode_max_steps
        self.current_step = 0
        self.total_reward = 0
        self.current_episode = 0  # Initialize episode counter

        # Create CSV files for logging and write headers
        if not os.path.exists(f'logs_{self.algorithm}'):
            os.makedirs(f'logs_{self.algorithm}')
        self.unique_id = int(time.time() * 1000)  # Timestamp in milliseconds
        step_file_name = f'logs_{self.algorithm}/steps_data_{self.unique_id}.csv'  # Log each step
        episode_file_name = f'logs_{self.algorithm}/episodes_data_{self.unique_id}.csv'  # Log each episode
        self.step_csv_file = open(step_file_name, mode='w', newline='')
        self.step_csv_writer = csv.writer(self.step_csv_file)
        self.step_csv_writer.writerow(['Step Time', 'Step', 'Target Temp', 'Actual Temp', 'Cooling Temp', 'Action', 'Heat DC', 'Cool DC', 'Dist DC', 'Last Reward', 'Previous Actual Temp', 'Previous Cooling Temp', 'Steps Within Target Range', 'Steps Outside Critical Range', 'Done'])
        self.episode_csv_file = open(episode_file_name, mode='w', newline='')
        self.episode_csv_writer = csv.writer(self.episode_csv_file)
        self.episode_csv_writer.writerow(['Episode Time', 'Episode', 'Initial Temp', 'Target Temp', 'Total Reward', 'Mean Reward per Step', 'Total Steps', 'First Within Target Step'])
        # Immediately flush files to ensure headers are written
        self.step_csv_file.flush()
        self.episode_csv_file.flush()

    def reset(self):
        self.current_episode += 1
        self.dist_dc = 0
        self.current_dist_steps = 0
            
        max_wait_time = 10  # Maximum wait time (seconds)
        wait_time = 0
        interval = 0.5  # Time interval for each check (seconds)

        # Try to obtain temperature data, avoiding zero readings due to potential issues on the Raspberry Pi side
        while wait_time < max_wait_time:
            self.actual_temp, self.cooling_temp = self.mqtt_client.get_temperature()
            if self.actual_temp != 0 and self.cooling_temp != 0:
                break  # Exit loop when valid temperature data is obtained
            time.sleep(interval)
            wait_time += interval

        # Raise an error if valid temperature data is not obtained within max_wait_time
        if self.actual_temp == 0 and self.cooling_temp == 0:
            raise ValueError("Failed to get temperature from MQTT client within the maximum wait time.")

        # Log the obtained temperature values
        print(f"reset: Fetched actual_temp={self.actual_temp}, cooling_temp={self.cooling_temp} from MQTT")
    
        # Determine target temperature based on obtained temperature data
        self.target_temp = EnvUtils.choose_target_temp(self.actual_temp, self.target_temp_min, self.target_temp_max, self.temp_config)
        
        # Log the determined target temperature
        print(f"reset: actual_temp={self.actual_temp}, target_temp={self.target_temp}")        
        
        # Initialize counters and buffers
        self.current_step = 0
        self.steps_within_target_range = 0
        self.steps_outside_critical_range = 0
        self.previous_actual_temp = None
        self.previous_cooling_temp = None   
        self.first_within_target_step = None  # Record the first step within the target temperature range in each episode
        self.total_reward = 0
        self.initial_temp = self.actual_temp  # Record initial temperature
        self.time_outside_target_range = 0
        
        return np.array([self.target_temp, self.actual_temp, self.cooling_temp]), {}

    def step(self, action=None):
        # Record the temperature data from the previous step
        self.previous_actual_temp = self.actual_temp
        self.previous_cooling_temp = self.cooling_temp    
        
        self.current_step += 1
        truncated = False

        if self.algorithm == 'Random':
            action_value = np.random.uniform(-1, 1) * 100  # For the Random method, randomly choose an action
        elif self.algorithm == 'Oven':  # On-Off Controller
            if self.actual_temp < self.target_temp:
                action_value = 100  # If the temperature is below the target, heat at full power
            else:
                action_value = -100  # If the temperature is above the target, cool at full power
        else:
            action_value = self.controller.compute(self.target_temp, self.actual_temp)  # Use Fuzzy or PID function

        # Convert action_value into PWM duty cycle data
        heat_dc, cool_dc = (0, abs(action_value)) if action_value < 0 else (abs(action_value), 0)

        # Disturbance module
        if self.dist_config and self.current_dist_steps == 0:
            if np.random.rand() < self.chance_of_zero:  # Decide whether to apply disturbance based on the probability of zero
                self.dist_dc = 0
            else:
                self.dist_dc = np.random.uniform(self.dist_config['dist_min'], self.dist_config['dist_max'])  # Randomly select a PWM signal value within the range
            self.current_dist_steps = self.dist_config['step_duration'] - 1
        elif self.current_dist_steps > 0:
            self.current_dist_steps -= 1  # Countdown for current_dist_steps to determine how many steps remain before changing the disturbance value 
        
        self.mqtt_client.publish_pwm(heat_dc, cool_dc, self.dist_dc)

        time.sleep(2)  # Delay to wait for the PWM signal to be executed
        
        # Get new temperature
        self.actual_temp, self.cooling_temp = self.mqtt_client.get_temperature()
        self.heat_dc = heat_dc
        self.cool_dc = cool_dc    
        
        # Update counters based on temperature, check if the temperature is within the range, see env_utils.py for details
        self.steps_within_target_range, self.steps_outside_critical_range, within_target_range = EnvUtils.update_temperature_counters(self.target_temp, self.actual_temp, self.steps_within_target_range, self.steps_outside_critical_range, self.temp_config)  
        
        # Record the first step within the target range
        if self.first_within_target_step is None and within_target_range:
            self.first_within_target_step = self.current_step

        # Use env_utils.py to calculate reward and done, and check if within the target range        
        reward, self.time_outside_target_range = EnvUtils.compute_reward(self.target_temp, self.actual_temp, self.time_outside_target_range, self.episode_max_steps, self.temp_config)
        self.last_reward = reward
        
        done = EnvUtils.check_done(self.steps_within_target_range, self.steps_outside_critical_range, self.temp_config)
        
        # Timeout logic
        if self.current_step >= self.episode_max_steps:
            truncated = True  # Truncated due to reaching max steps

        # Update observation data
        info = {}  # Can be used to add additional debug or state information
        observation = np.array([self.target_temp, self.actual_temp, self.cooling_temp])

        # Get timestamp
        self.unique_id = int(time.time() * 1000)
        
        print(f'Time {self.unique_id}: Step {self.current_step:03}: Target: {self.target_temp}, Actual: {self.actual_temp:.3f}, Cooling: {self.cooling_temp:.3f}, Heat DC: {self.heat_dc:07.3f}, Cool DC: {self.cool_dc:07.3f}, Dist DC: {self.dist_dc:07.3f}, Reward: {self.last_reward:.3f}, Done: {done}')

        # Log current step to CSV
        self.step_csv_writer.writerow([self.unique_id, self.current_step, self.target_temp, self.actual_temp, self.cooling_temp, action_value, self.heat_dc, self.cool_dc, self.dist_dc, self.last_reward, self.previous_actual_temp, self.previous_cooling_temp, self.steps_within_target_range, self.steps_outside_critical_range, done])
        self.step_csv_file.flush()

        # Update total reward and step count
        self.total_reward += reward

        if done or truncated:
            # Log episode results to CSV
            mean_reward = self.total_reward / self.current_step if self.current_step != 0 else 0
            self.unique_id = int(time.time() * 1000)
            self.episode_csv_writer.writerow([self.unique_id, self.current_episode, self.initial_temp, self.target_temp, self.total_reward, mean_reward, self.current_step, self.first_within_target_step])
            self.episode_csv_file.flush()

        return observation, reward, done, truncated, info        

    def close(self):
        """Terminate the environment"""
        super().close()
        self.step_csv_file.flush()
        self.step_csv_file.close()
        self.episode_csv_file.flush()
        self.episode_csv_file.close()
        self.mqtt_client.disconnect()  # Disconnect from the MQTT broker

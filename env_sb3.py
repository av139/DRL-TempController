# env_sb3.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from mqtt_win import MQTTClient
import csv
import random
import os
import time
from env_utils import EnvUtils

# Define a custom environment that extends the gym.Env class, used for handling DRL algorithms in SB3
class SB3Env(gym.Env):
    def __init__(self, broker_address, port, temp_topic, pwm_topic, target_temp_min, target_temp_max, algorithm, action_space_type='box', episode_max_steps=1000, render_mode='human', discrete_step=10, temp_config=None, dist_config=None):
    
        super(SB3Env, self).__init__()
        
        self.temp_config = temp_config
        self.render_mode = render_mode
        self.algorithm = algorithm
        self.action_space_type = action_space_type
        self.discrete_step = discrete_step
        
        self.dist_config = dist_config
        self.dist_dc = 0
        self.current_dist_steps = 0
        self.chance_of_zero = dist_config.get('chance_of_zero', 0) if dist_config else 0
                
        # Select action space based on action_space_type
        if action_space_type == 'box':
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        elif action_space_type == 'discrete':
            self.action_space = spaces.Discrete(200 // self.discrete_step + 1)
        else:
            raise ValueError(f"Unsupported action_space_type: {action_space_type}")
        
        # Set observation space, which represents the output range of the temperature sensors: target, actual on the object, and cooling temperature        
        self.observation_space = spaces.Box(
            low=np.array([self.temp_config['target_temp_low'], self.temp_config['actual_temp_low'], self.temp_config['cooling_temp_low']], dtype=np.float32),
            high=np.array([self.temp_config['target_temp_high'], self.temp_config['actual_temp_high'], self.temp_config['cooling_temp_high']], dtype=np.float32)
        )

        self.target_temp_min = target_temp_min
        self.target_temp_max = target_temp_max

        # Episode steps limit
        self.episode_max_steps = episode_max_steps
        self.current_step = 0

        # MQTT setup
        self.mqtt_client = MQTTClient(broker_address, port, temp_topic, pwm_topic)
        self.mqtt_client.connect()
        
        self.steps_within_target_range = 0  # Counter for tracking steps within target range for done checking based on time steps
        self.steps_outside_critical_range = 0  # Counter for tracking steps outside critical range for done checking based on time steps
        self.recovery_flag = False
        self.last_reward = 0.0  # Initialize last reward value
        self.current_episode = 0  # Initialize episode counter
        self.first_within_target_step = None  # Record the first step within the target temperature range
        self.time_outside_target_range = 0  # Counter for tracking time outside the target range for reward calculation
        
        # Create CSV files for logging and write headers
        if not os.path.exists(f'logs_{self.algorithm}'):
            os.makedirs(f'logs_{self.algorithm}')
        self.unique_id = int(time.time() * 1000)    
        step_file_name = f'logs_{self.algorithm}/steps_data_{self.algorithm}_{self.action_space_type}_{self.unique_id}.csv'  # Log each step
        episode_file_name = f'logs_{self.algorithm}/episodes_data_{self.algorithm}_{self.action_space_type}_{self.unique_id}.csv'  # Log each episode
        # Write headers
        self.step_csv_file = open(step_file_name, mode='w', newline='')
        self.step_csv_writer = csv.writer(self.step_csv_file)
        self.step_csv_writer.writerow(['Step Time', 'Step', 'Target Temp', 'Actual Temp', 'Cooling Temp', 'Action', 'Heat DC', 'Cool DC', 'Dist DC', 'Last Reward', 'Previous Actual Temp', 'Previous Cooling Temp', 'Steps Within Target Range', 'Steps Outside Critical Range', 'Done'])
        self.episode_csv_file = open(episode_file_name, mode='w', newline='')
        self.episode_csv_writer = csv.writer(self.episode_csv_file)
        self.episode_csv_writer.writerow(['Episode Time', 'Episode', 'Initial Temp', 'Target Temp', 'Total Reward', 'Mean Reward per Step', 'Total Steps', 'First Within Target Step'])
        
        # Immediately flush files to ensure headers are written
        self.step_csv_file.flush()
        self.episode_csv_file.flush()        


    def reset(self, seed=None, **kwargs):
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


    def step(self, action):
        # Save previous temperatures
        self.previous_actual_temp = self.actual_temp
        self.previous_cooling_temp = self.cooling_temp    
        
        # Step counter for each episode 
        self.current_step += 1
        
        # Timeout flag
        truncated = False
        
        # Check if self.action_space is an instance of spaces.Discrete
        # Discrete and continuous tasks map actions to PWM signals differently, so this needs to be distinguished
        # Map the action signal to the range -1 to +1
        if isinstance(self.action_space, spaces.Discrete):
            action_value = (action - (200 // self.discrete_step // 2)) / (200 // self.discrete_step // 2)  # Discrete action space
        else:
            action_value = action[0]  # Continuous action space
            
        # Then convert the signal in the range -1 to +1 to PWM signals in the range -100% to +100%, and assign them to heating/cooling elements
        # Heating and cooling elements do not work simultaneously
        heat_dc, cool_dc = (0, abs(action_value) * 100) if action_value < 0 else (abs(action_value) * 100, 0)
        
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

        # Output
        info = {}  # Can be used to add additional debug or state information
        observation = np.array([self.target_temp, self.actual_temp, self.cooling_temp])

        # Get timestamp
        self.unique_id = int(time.time() * 1000)
        
        print(f'Time {self.unique_id}: Step {self.current_step:03}: Target: {self.target_temp}, Actual: {self.actual_temp:.3f}, Cooling: {self.cooling_temp:.3f}, Action: {action_value:07.3f}, Heat DC: {self.heat_dc:07.3f}, Cool DC: {self.cool_dc:07.3f}, Dist DC: {self.dist_dc:07.3f}, Reward: {self.last_reward:.3f}, Done: {done}')

        # Log current step to CSV
        self.step_csv_writer.writerow([self.unique_id, self.current_step, self.target_temp, self.actual_temp, self.cooling_temp, action_value, self.heat_dc, self.cool_dc, self.dist_dc, self.last_reward, self.previous_actual_temp, self.previous_cooling_temp, self.steps_within_target_range, self.steps_outside_critical_range, done])
        self.step_csv_file.flush()

        # Update total reward and step count
        self.total_reward += reward

        if done or truncated:
            # Log episode results to CSV
            # Calculate the mean reward per step for the episode
            mean_reward = self.total_reward / self.current_step if self.current_step != 0 else 0
            # Record the end time of the episode
            self.unique_id = int(time.time() * 1000)
            # Episode Time, Episode, Initial Temp, Target Temp, Total Reward, Mean Reward per Step, Total Steps, First Within Target Step
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

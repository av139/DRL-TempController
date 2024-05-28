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
        error = target_temp - actual_temp
        self.integral += error
        self.derivative = error - self.previous_error
        self.previous_error = error

        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * self.derivative)
        output = max(min(output, 100), -100)
        return output

class FuzzyController:
    def __init__(self):
        # 定义模糊变量
        self.error = ctrl.Antecedent(np.arange(-30, 31, 1), 'error')
        self.output = ctrl.Consequent(np.arange(-100, 101, 1), 'output')

        # 定义模糊集合
        self.error['negative'] = fuzz.trimf(self.error.universe, [-30, -5, 0])
        self.error['zero'] = fuzz.trimf(self.error.universe, [-5, 0, 5])
        self.error['positive'] = fuzz.trimf(self.error.universe, [0, 5, 30])

        self.output['cool'] = fuzz.trimf(self.output.universe, [-100, -100, 0])
        self.output['zero'] = fuzz.trimf(self.output.universe, [-100, 0, 100])
        self.output['heat'] = fuzz.trimf(self.output.universe, [0, 100, 100])

        # 定义规则
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
        self.current_episode = 0  # 初始化episode计数器

        # CSV
        if not os.path.exists(f'logs_{self.algorithm}'):
            os.makedirs(f'logs_{self.algorithm}')
        self.unique_id = int(time.time() * 1000)
        step_file_name = f'logs_{self.algorithm}/steps_data_{self.unique_id}.csv'
        episode_file_name = f'logs_{self.algorithm}/episodes_data_{self.unique_id}.csv'
        self.step_csv_file = open(step_file_name, mode='w', newline='')
        self.step_csv_writer = csv.writer(self.step_csv_file)
        self.step_csv_writer.writerow(['Step Time', 'Step', 'Target Temp', 'Actual Temp', 'Cooling Temp', 'Action', 'Heat DC', 'Cool DC', 'Dist DC', 'Last Reward', 'Previous Actual Temp', 'Previous Cooling Temp', 'Steps Within Target Range', 'Steps Outside Critical Range', 'Done'])
        self.episode_csv_file = open(episode_file_name, mode='w', newline='')
        self.episode_csv_writer = csv.writer(self.episode_csv_file)
        self.episode_csv_writer.writerow(['Episode Time', 'Episode', 'Initial Temp', 'Target Temp', 'Total Reward', 'Mean Reward per Step', 'Total Steps', 'First Within Target Step'])
        # 立即刷新文件，确保表头被写入
        self.step_csv_file.flush()
        self.episode_csv_file.flush()

    def reset(self):
        self.current_episode += 1
        self.dist_dc = 0
        self.current_dist_steps = 0
            
        # 尝试获取温度数据，增加等待逻辑
        max_wait_time = 10  # 最大等待时间（秒）
        wait_time = 0
        interval = 0.5  # 每次检查的间隔时间（秒）

        while wait_time < max_wait_time:
            self.actual_temp, self.cooling_temp = self.mqtt_client.get_temperature()
            if self.actual_temp != 0 and self.cooling_temp != 0:
                break
            time.sleep(interval)
            wait_time += interval

        if self.actual_temp == 0 and self.cooling_temp == 0:
            raise ValueError("Failed to get temperature from MQTT client within the maximum wait time.")

        # 记录日志，确认获取到的温度值
        print(f"reset: Fetched actual_temp={self.actual_temp}, cooling_temp={self.cooling_temp} from MQTT")
    
        self.target_temp = EnvUtils.choose_target_temp(self.actual_temp, self.target_temp_min, self.target_temp_max, self.temp_config)
        
        # 记录日志
        print(f"reset: actual_temp={self.actual_temp}, target_temp={self.target_temp}")        
        
        # Initialize counters and buffers
        self.current_step = 0
        self.steps_within_target_range = 0
        self.steps_outside_critical_range = 0
        self.previous_actual_temp = None
        self.previous_cooling_temp = None   
        self.first_within_target_step = None
        self.total_reward = 0
        self.initial_temp = self.actual_temp  # 记录初始温度
        self.time_outside_target_range = 0
        
        return np.array([self.target_temp, self.actual_temp, self.cooling_temp]), {}

    def step(self, action=None):
        self.previous_actual_temp = self.actual_temp
        self.previous_cooling_temp = self.cooling_temp    
        
        self.current_step += 1
        truncated = False

        if self.algorithm == 'Random':
            action_value = np.random.uniform(-1, 1) * 100
        else:
            action_value = self.controller.compute(self.target_temp, self.actual_temp)

        heat_dc, cool_dc = (0, abs(action_value)) if action_value < 0 else (abs(action_value), 0)

        if self.dist_config and self.current_dist_steps == 0:
            if np.random.rand() < self.chance_of_zero:
                self.dist_dc = 0
            else:
                self.dist_dc = np.random.uniform(self.dist_config['dist_min'], self.dist_config['dist_max'])
            self.current_dist_steps = self.dist_config['step_duration'] - 1
        elif self.current_dist_steps > 0:
            self.current_dist_steps -= 1      
        
        self.mqtt_client.publish_pwm(heat_dc, cool_dc, self.dist_dc)

        time.sleep(2)  # Delay to simulate response time
        
        # get new temp
        self.actual_temp, self.cooling_temp = self.mqtt_client.get_temperature()
        self.heat_dc = heat_dc
        self.cool_dc = cool_dc    
        
        # 调用env_utils.py来计算reward和done，以及检查是否在目标范围内
        self.steps_within_target_range, self.steps_outside_critical_range, within_target_range = EnvUtils.update_temperature_counters(self.target_temp, self.actual_temp, self.steps_within_target_range, self.steps_outside_critical_range, self.temp_config)  
        
        # 记录第一次进入目标范围内的步数
        if self.first_within_target_step is None and within_target_range:
            self.first_within_target_step = self.current_step
        
        reward, self.time_outside_target_range = EnvUtils.compute_reward(self.target_temp, self.actual_temp, self.time_outside_target_range, self.episode_max_steps, self.temp_config)
        self.last_reward = reward
        
        done = EnvUtils.check_done(self.steps_within_target_range, self.steps_outside_critical_range, self.temp_config)
        
        # timeout 逻辑判定
        if self.current_step >= self.episode_max_steps:
            truncated = True  # Truncated due to reaching max steps

        # action_value
        info = {}  # 可以添加额外的调试信息或状态信息
        observation = np.array([self.target_temp, self.actual_temp, self.cooling_temp])

        # 获取时间戳
        self.unique_id = int(time.time() * 1000)
        
        print(f'Time {self.unique_id}: Step {self.current_step:03}: Target: {self.target_temp}, Actual: {self.actual_temp:.3f}, Cooling: {self.cooling_temp:.3f}, Heat DC: {self.heat_dc:07.3f}, Cool DC: {self.cool_dc:07.3f}, Dist DC: {self.dist_dc:07.3f}, Reward: {self.last_reward:.3f}, Done: {done}')

        # 记录当前步骤到CSV
        self.step_csv_writer.writerow([self.unique_id, self.current_step, self.target_temp, self.actual_temp, self.cooling_temp, action_value, self.heat_dc, self.cool_dc, self.dist_dc, self.last_reward, self.previous_actual_temp, self.previous_cooling_temp, self.steps_within_target_range, self.steps_outside_critical_range, done])
        self.step_csv_file.flush()

        # 更新奖励总和和步数
        self.total_reward += reward

        if done or truncated:
            # 记录episode结果到CSV
            mean_reward = self.total_reward / self.current_step if self.current_step != 0 else 0
            self.unique_id = int(time.time() * 1000)
            self.episode_csv_writer.writerow([self.unique_id, self.current_episode, self.initial_temp, self.target_temp, self.total_reward, mean_reward, self.current_step, self.first_within_target_step])
            self.episode_csv_file.flush()

        return observation, reward, done, truncated, info        

    def close(self):
        """结束环境"""
        super().close()
        self.step_csv_file.flush()
        self.step_csv_file.close()
        self.episode_csv_file.flush()
        self.episode_csv_file.close()
        self.mqtt_client.disconnect()  # 断开与MQTT服务器的连接

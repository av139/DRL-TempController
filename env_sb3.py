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

# 定义一个自定义环境，继承gym.Env类
class SB3Env(gym.Env):
    def __init__(self, broker_address, port, temp_topic, pwm_topic, target_temp_min, target_temp_max, algorithm, action_space_type='box', episode_max_steps=1000, render_mode='human', discrete_step=10, temp_config=None):
    
        super(SB3Env, self).__init__()
        
        self.temp_config = temp_config
        self.render_mode = render_mode
        self.algorithm = algorithm
        self.discrete_step = discrete_step
        
        self.dist_config = dist_config
        self.dist_dc = 0
        self.current_dist_steps = 0
        self.chance_of_zero = dist_config.get('chance_of_zero', 0) if dist_config else 0
                
        # 根据action_space_type选择动作空间
        if action_space_type == 'box':
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        elif action_space_type == 'discrete':
            self.action_space = spaces.Discrete(200 // self.discrete_step + 1)
        else:
            raise ValueError(f"Unsupported action_space_type: {action_space_type}")
        
        # 设置观察空间，也就是温度传感器的输出范围，第一个值是target，第二个值是物体上的actual，第三个值是散热器cooling温度        
        self.observation_space = spaces.Box(
            low=np.array([self.temp_config['target_temp_low'], self.temp_config['actual_temp_low'], self.temp_config['cooling_temp_low']], dtype=np.float32),
            high=np.array([self.temp_config['target_temp_high'], self.temp_config['actual_temp_high'], self.temp_config['cooling_temp_high']], dtype=np.float32)
        )

        self.target_temp_min = target_temp_min
        self.target_temp_max = target_temp_max

        # episode steps limit
        self.episode_max_steps = episode_max_steps
        self.current_step = 0

        # MQTT
        self.mqtt_client = MQTTClient(broker_address, port, temp_topic, pwm_topic)
        self.mqtt_client.connect()
        
        self.steps_within_target_range = 0  # 给基于time steps的check done表达的计数器
        self.steps_outside_critical_range = 0   # 给基于time steps的check done表达的计数器
        self.recovery_flag = False
        self.last_reward = 0.0  # 初始化最新奖励值
        self.current_episode = 0  # 初始化episode计数器
        self.first_within_target_step = None  # 记录第一次实际温度在目标温度范围内的步数
        self.time_outside_target_range = 0  # 给基于time steps的reward表达的计数器
        
        # CSV
        if not os.path.exists(f'logs_{self.algorithm}'):
            os.makedirs(f'logs_{self.algorithm}')
        self.unique_id = int(time.time() * 1000)    
        step_file_name = f'logs_{self.algorithm}/steps_data_{self.algorithm}_{self.action_space_type}_{self.unique_id}.csv'
        episode_file_name = f'logs_{self.algorithm}/episodes_data_{self.algorithm}_{self.action_space_type}_{self.unique_id}.csv'
        self.step_csv_file = open(step_file_name, mode='w', newline='')
        self.step_csv_writer = csv.writer(self.step_csv_file)
        self.step_csv_writer.writerow(['Step Time', 'Step', 'Target Temp', 'Actual Temp', 'Cooling Temp', 'Heat DC', 'Cool DC', 'Dist DC', 'Last Reward', 'Previous Actual Temp', 'Previous Cooling Temp', 'Steps Within Target Range', 'Steps Outside Critical Range', 'Done'])
        self.episode_csv_file = open(episode_file_name, mode='w', newline='')
        self.episode_csv_writer = csv.writer(self.episode_csv_file)
        self.episode_csv_writer.writerow(['Episode Time', 'Episode', 'Initial Temp', 'Target Temp', 'Total Reward', 'Mean Reward per Step', 'Total Steps', 'First Within Target Step'])
        
        # 立即刷新文件，确保表头被写入
        self.step_csv_file.flush()
        self.episode_csv_file.flush()        


    def reset(self, seed=None, **kwargs):
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
        
        # reset the target_temp
        self.target_temp = EnvUtils.choose_target_temp(self.actual_temp, self.target_temp_min, self.target_temp_max, self.temp_config)
        
        # 记录日志
        print(f"reset: actual_temp={self.actual_temp}, target_temp={self.target_temp}") 

        # Initialize counters and buffers
        self.current_step = 0
        self.steps_within_target_range = 0
        self.steps_outside_critical_range = 0
        self.previous_actual_temp = None
        self.previous_cooling_temp = None   
        self.first_within_target_step = None  # 记录每个episode内第一次进入target temp范围内的步数
        self.total_reward = 0
        self.initial_temp = self.actual_temp  # 记录初始温度
        self.time_outside_target_range = 0
        
        return np.array([self.target_temp, self.actual_temp, self.cooling_temp]), {}


    def step(self, action):
        # save old temp
        self.previous_actual_temp = self.actual_temp
        self.previous_cooling_temp = self.cooling_temp    
        
        # step counter in each episode 
        self.current_step += 1
        
        # time out flag
        truncated = False
        
        # 用于检查 self.action_space 是否是 spaces.Discrete 类型的实例
        # 离散和连续任务有不同的action映射到PWM信号的方法，需要区分
        # 将action信号首先映射到 -1 ~ +1区间
        if isinstance(self.action_space, spaces.Discrete):
            action_value = (action - (200 // self.discrete_step // 2)) / (200 // self.discrete_step // 2) # 当是Discrete，离散动作
        else:
            action_value = action[0] # 当是Box，连续动作
            
        # 再将位于-1 ~ +1区间的信号转换成-100% ~ +100%的PWM信号，并分配给加热/散热元件
        # 加热和散热元件不同时工作
        heat_dc, cool_dc = (0, abs(action_value) * 100) if action_value < 0 else (abs(action_value) * 100, 0)
        
        if self.dist_config and self.current_dist_steps == 0:
            if np.random.rand() < self.chance_of_zero:
                self.dist_dc = 0
            else:
                self.dist_dc = np.random.uniform(self.dist_config['dist_min'], self.dist_config['dist_max'])
            self.current_dist_steps = self.dist_config['step_duration'] - 1
        elif self.current_dist_steps > 0:
            self.current_dist_steps -= 1    
        
        self.mqtt_client.publish_pwm(heat_dc, cool_dc, self.dist_dc)
        """这里缺少将负数的dist pwm信号剪切的逻辑"""
        time.sleep(2)  # 延时2秒，等待温度变化

        # get new temp
        self.actual_temp, self.cooling_temp = self.mqtt_client.get_temperature()
        self.heat_dc = heat_dc
        self.cool_dc = cool_dc     

        # counter_update & reward & done
        # 调用env_utils.py来计算reward和done，以及检查是否在target_temp范围内
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

        # output
        info = {}  # 可以添加额外的调试信息或状态信息
        observation = np.array([self.target_temp, self.actual_temp, self.cooling_temp])

        # 获取时间戳
        self.unique_id = int(time.time() * 1000)
        
        print(f'Time {self.unique_id}: Step {self.current_step:03}: Target: {self.target_temp}, Actual: {self.actual_temp:.3f}, Cooling: {self.cooling_temp:.3f}, Heat DC: {self.heat_dc:07.3f}, Cool DC: {self.cool_dc:07.3f}, Dist DC: {self.dist_dc:07.3f}, Reward: {self.last_reward:.3f}, Done: {done}')

        # 记录当前步骤到CSV
        self.step_csv_writer.writerow([self.unique_id, self.current_step, self.target_temp, self.actual_temp, self.cooling_temp, self.heat_dc, self.cool_dc, self.dist_dc, self.last_reward, self.previous_actual_temp, self.previous_cooling_temp, self.steps_within_target_range, self.steps_outside_critical_range, done])
        self.step_csv_file.flush()

        # 更新奖励总和和步数
        self.total_reward += reward

        if done or truncated:
            # 记录episode结果到CSV
            # 计算每个episode周期内的单步平均奖励值
            mean_reward = self.total_reward / self.current_step if self.current_step != 0 else 0
            # 记录每个episode的结束时间
            self.unique_id = int(time.time() * 1000)
            # Episode Time, Episode, Initial Temp, Target Temp, Total Reward, Mean Reward per Step, Total Steps, First Within Target Step
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

#env_utils.py
import numpy as np
import random

class EnvUtils:
    @staticmethod
    def compute_reward(target_temp, actual_temp, time_outside_target_range, episode_max_steps, config):
        """奖励模式：温差reward + 时间reward"""
        max_diff = config['max_diff']
        temp_tolerance = config['temp_tolerance']
        # 温差reward
        temp_diff = abs(target_temp - actual_temp)
        temp_reward = -temp_diff / max_diff
        
        # 时间reward
        if abs(actual_temp - target_temp) <= temp_tolerance:
            time_reward = 0
        else:
            time_outside_target_range += 1
            time_reward = -time_outside_target_range / episode_max_steps
        
        # 总reward
        total_reward = 0.6 * temp_reward + 0.4 * time_reward
        return total_reward, time_outside_target_range

    @staticmethod
    def check_done(steps_within_target_range, steps_outside_critical_range, config):
        """检查是否完成目标"""
        critical_range_steps = config['critical_range_steps']
        if steps_outside_critical_range >= critical_range_steps:
            return True
        if steps_within_target_range >= critical_range_steps:
            return True
        return False

    @staticmethod
    def update_temperature_counters(target_temp, actual_temp, steps_within_target_range, steps_outside_critical_range, config):
        """更新温度相关的计数器状态"""
        temp_tolerance = config['temp_tolerance']
        max_diff = config['max_diff']
        within_target_range = False
        if abs(actual_temp - target_temp) <= temp_tolerance:
            steps_within_target_range += 1
            within_target_range = True
        else:
            steps_within_target_range = 0
            # within_target_range = False

        if actual_temp > target_temp + max_diff or actual_temp < target_temp - max_diff:
            steps_outside_critical_range += 1
        else:
            steps_outside_critical_range = 0

        return steps_within_target_range, steps_outside_critical_range, within_target_range


    @staticmethod
    def choose_target_temp(actual_temp, target_temp_min, target_temp_max, config):
        """根据当前温度选择目标温度"""
        max_diff = config['max_diff']
        target_temp_range = config['target_temp_range']

        # Define the target temperature ranges based on current temperature
        low_range = max(actual_temp - max_diff, target_temp_min)
        high_range = min(actual_temp + max_diff, target_temp_max)

        # Define the valid range
        valid_ranges = []
        if low_range <= actual_temp - target_temp_range:
            valid_ranges.append((low_range, actual_temp - target_temp_range))
        if high_range >= actual_temp + target_temp_range:
            valid_ranges.append((actual_temp + target_temp_range, high_range))

        # Find the intersection of valid ranges with target temp min and max
        intersected_ranges = []
        for range_start, range_end in valid_ranges:
            start = max(range_start, target_temp_min)
            end = min(range_end, target_temp_max)
            if start <= end:
                intersected_ranges.append((start, end))

        # Choose a target temperature from the intersected valid ranges
        if intersected_ranges:
            chosen_range = random.choice(intersected_ranges)
            target_temp = round(random.uniform(chosen_range[0], chosen_range[1]), 1)
        else:
            # Emergency fallback if no valid range
            target_temp = round(random.uniform(target_temp_min, target_temp_max), 1)  

        # 增加日志记录
        # print(f"choose_target_temp: actual_temp={actual_temp}, target_temp_min={target_temp_min}, target_temp_max={target_temp_max}")
        # print(f"choose_target_temp: low_range={low_range}, high_range={high_range}")
        # print(f"choose_target_temp: valid_ranges={valid_ranges}")
        # print(f"choose_target_temp: intersected_ranges={intersected_ranges}")
        # print(f"choose_target_temp: chosen target_temp={target_temp}")

        return target_temp


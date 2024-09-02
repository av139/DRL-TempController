# env_utils.py
import numpy as np
import random

class EnvUtils:
    @staticmethod
    def compute_reward(target_temp, actual_temp, time_outside_target_range, episode_max_steps, config):
        """Reward scheme: Temperature difference reward + Time reward"""
        max_diff = config['max_diff']
        temp_tolerance = config['temp_tolerance']
        
        # Temperature difference reward
        temp_diff = abs(target_temp - actual_temp)
        temp_reward = -temp_diff / max_diff
        
        # Time reward
        if abs(actual_temp - target_temp) <= temp_tolerance:
            time_reward = 0
        else:
            # This commented-out line is a modified reward method; see section 6.2 of the paper
            # time_reward = -1 
            # ----------------------------------
            # The following is the original TOR reward method
            time_outside_target_range += 1
            time_reward = -time_outside_target_range / episode_max_steps
        
        # Total reward
        # Note: Weighted weights need to be assigned here. This weight is not created in the config and should be adjusted based on the actual reward conditions
        total_reward = 0.6 * temp_reward + 0.4 * time_reward
        return total_reward, time_outside_target_range

    @staticmethod
    def check_done(steps_within_target_range, steps_outside_critical_range, config):
        """Check if the target is achieved"""
        critical_range_steps = config['critical_range_steps']
        if steps_outside_critical_range >= critical_range_steps:
            return True
        if steps_within_target_range >= critical_range_steps:
            return True
        return False

    @staticmethod
    def update_temperature_counters(target_temp, actual_temp, steps_within_target_range, steps_outside_critical_range, config):
        """Update the counters related to temperature status"""
        temp_tolerance = config['temp_tolerance']
        max_diff = config['max_diff']
        within_target_range = False
        
        # Check if the current temperature is within the target range
        if abs(actual_temp - target_temp) <= temp_tolerance:
            steps_within_target_range += 1
            within_target_range = True
        else:
            steps_within_target_range = 0

        # Check if the temperature is far outside the allowable deviation range
        if actual_temp > target_temp + max_diff or actual_temp < target_temp - max_diff:
            steps_outside_critical_range += 1
        else:
            steps_outside_critical_range = 0

        return steps_within_target_range, steps_outside_critical_range, within_target_range

    @staticmethod
    def choose_target_temp(actual_temp, target_temp_min, target_temp_max, config):
        max_diff = config['max_diff']
        target_temp_range = config['target_temp_range']

        # Define the target temperature ranges based on the current temperature
        low_range = max(actual_temp - max_diff, target_temp_min)
        high_range = min(actual_temp + max_diff, target_temp_max)

        # Define the valid range
        valid_ranges = []
        if low_range <= actual_temp - target_temp_range:
            valid_ranges.append((low_range, actual_temp - target_temp_range))
        if high_range >= actual_temp + target_temp_range:
            valid_ranges.append((actual_temp + target_temp_range, high_range))

        # Find the intersection of valid ranges with target temperature min and max
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
            # Emergency fallback if no valid range exists
            target_temp = round(random.uniform(target_temp_min, target_temp_max), 1)  

        return target_temp

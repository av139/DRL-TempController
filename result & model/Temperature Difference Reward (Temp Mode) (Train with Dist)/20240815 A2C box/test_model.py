# test_model.py

import argparse
from stable_baselines3 import SAC, PPO, DQN, A2C, DDPG, TD3
from sb3_contrib import TRPO
import yaml
from env_sb3 import SB3Env
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# 解析命令行参数
def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate pre-trained model")
    parser.add_argument('--model_path', type=str, help="Path to the model zip file", required=True)
    parser.add_argument('--algorithm', type=str, help="Algorithm type", required=True)
    parser.add_argument('--action_space_type', type=str, default="box", help="Type of action space ('box' or 'discrete')")
    parser.add_argument('--episode_max_steps', type=int, default=500, help="Maximum number of steps per episode")
    parser.add_argument('--n_eval_episodes', type=int, default=5, help="Number of evaluation episodes")
    return parser.parse_args()

# 调整动作空间类型以确保兼容性
def adjust_action_space(algorithm, action_space_type):
    corrected_type = action_space_type
    if algorithm in ['SAC', 'DDPG', 'TD3']:
        if action_space_type != 'box':
            corrected_type = 'box'
            print(f"Warning: {algorithm} only supports 'box' action space. Adjusting action_space_type to 'box'.")
    elif algorithm == 'DQN':
        if action_space_type != 'discrete':
            corrected_type = 'discrete'
            print("Warning: DQN only supports 'discrete' action space. Adjusting action_space_type to 'discrete'.")
    elif algorithm in ['PPO', 'A2C', 'TRPO']:
        if action_space_type not in ['box', 'discrete']:
            corrected_type = 'box'  # Default to 'box'
            print(f"Warning: {algorithm} supports both 'box' and 'discrete' action spaces. Defaulting to 'box'.")
    return corrected_type

# 加载和配置环境
def load_env(config, algorithm, action_space_type, episode_max_steps):
    general_config = config['general']
    specific_config = config['dist_config_test_model']
    temp_config = config.get('temp_config')
    
    env = SB3Env(
        broker_address=general_config['broker_address'],
        port=general_config['port'],
        temp_topic=general_config['temp_topic'],
        pwm_topic=general_config['pwm_topic'],
        target_temp_min=general_config['target_temp_min'],
        target_temp_max=general_config['target_temp_max'],
        render_mode='human',
        algorithm=algorithm,
        action_space_type=action_space_type,
        episode_max_steps=episode_max_steps,
        discrete_step=general_config.get('discrete_step', 10),
        temp_config=temp_config,
        dist_config=specific_config
    )
    return env

# 根据算法类型加载相应的模型
def load_model(model_path, algorithm):
    if algorithm == 'TRPO':
        model = TRPO.load(model_path)
    elif algorithm == 'SAC':
        model = SAC.load(model_path)
    elif algorithm == 'PPO':
        model = PPO.load(model_path)
    elif algorithm == 'DQN':
        model = DQN.load(model_path)
    elif algorithm == 'A2C':
        model = A2C.load(model_path)
    elif algorithm == 'DDPG':
        model = DDPG.load(model_path)
    elif algorithm == 'TD3':
        model = TD3.load(model_path)
    else:
        raise ValueError("Unsupported algorithm")
    return model

# 使用环境评估模型
def evaluate_model(model, env, n_eval_episodes):
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")
    return mean_reward, std_reward

# 主函数
def main():
    args = parse_arguments()
    env = None
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        corrected_action_space_type = adjust_action_space(args.algorithm, args.action_space_type)
        
        env = load_env(config, args.algorithm, corrected_action_space_type, args.episode_max_steps)
        env = Monitor(env)  # 包装评估环境
        env = DummyVecEnv([lambda: env])
        model = load_model(args.model_path, args.algorithm)
        
        # 更新日志目录以包括动作空间类型
        log_dir_suffix = f"{args.algorithm}_{corrected_action_space_type}" if args.algorithm in ['PPO', 'A2C', 'TRPO'] else args.algorithm
        log_dir = f"./tensorboard/test_model/{log_dir_suffix}/"
        
        with SummaryWriter(log_dir=log_dir) as writer:
            mean_reward, std_reward = evaluate_model(model, env, n_eval_episodes=args.n_eval_episodes)
            writer.add_scalar('test_model/mean_reward', mean_reward, 0)
            writer.add_scalar('test_model/std_reward', std_reward, 0)

    except KeyboardInterrupt:
        print("Evaluation interrupted by user.")
    finally:
        if env is not None:
            env.close()  # Ensure environment is closed properly
        print("Environment has been safely closed.")

if __name__ == "__main__":
    main()

# python test_model.py --model_path path_to_your_model/your_model.zip --algorithm YourAlgorithm --action_space_type continuous --episode_max_steps 800 --n_eval_episodes 10

# python test_model.py --model_path path_to_your_model/sac_model.zip --algorithm SAC --action_space_type box --episode_max_steps 800 --n_eval_episodes 10



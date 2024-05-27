# train.py
import yaml
from stable_baselines3 import SAC, PPO, DQN, A2C, DDPG, TD3
from sb3_contrib import TRPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from env_utils import EnvUtils
from env_customized import CustomEnv  
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from env_sb3 import SB3Env

def create_sb3_env(general_config, specific_config, algorithm, action_space_type, temp_config, dist_config, is_eval=False):
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
        episode_max_steps=specific_config['episode_max_steps'],
        discrete_step=general_config.get('discrete_step', 10),
        temp_config=temp_config,
        dist_config=dist_config if is_eval else None  # 在训练时dist_config设置为None
    )
    return env

# 新增函数：记录Random/Fuzzy/PID的评估数据到TensorBoard
def log_to_tensorboard(writer, tag, scalar_value, global_step):
    writer.add_scalar(tag, scalar_value, global_step)

# 用于进行random/fuzzy/pid三种方法的评估阶段
# 这里虽然传入了dist_config但好像没有用到
def evaluate_env(env, writer, total_timesteps, n_eval_episodes, dist_config):
    timesteps = 0
    try:
        while timesteps < total_timesteps:
            episode_lengths = []
            episode_rewards = []

            for _ in range(n_eval_episodes):
                print("New Episode")
                obs, _ = env.reset()
                done = False
                episode_length = 0
                episode_reward = 0

                while not done and timesteps < total_timesteps:
                    obs, reward, done, truncated, info = env.step()
                    timesteps += 1
                    episode_length += 1
                    episode_reward += reward

                    if done or truncated:
                        if done:
                            print("Episode finished with done = True")
                        if truncated:
                            print("Episode finished with truncated = True (timeout)")
                        break

                episode_lengths.append(episode_length)
                episode_rewards.append(episode_reward)

            mean_ep_length = np.mean(episode_lengths)
            mean_reward = np.mean(episode_rewards)
            writer.add_scalar('eval/mean_ep_length', mean_ep_length, timesteps)
            writer.add_scalar('eval/mean_reward', mean_reward, timesteps)
    except KeyboardInterrupt:
        print("Evaluation interrupted by user.")
        raise  # 重新引发异常，以便外部捕获

def run_random(config):
    env = None
    try:
        random_config = config['random']
        general_config = config['general']
        temp_config = config['temp_config']
        dist_config = config['dist_config']
        total_timesteps = random_config['total_timesteps']
        n_eval_episodes = general_config.get('n_eval_episodes', 5)  # 新增：评估的episode数量
        
        env = CustomEnv(
            broker_address=general_config['broker_address'],
            port=general_config['port'],
            temp_topic=general_config['temp_topic'],
            pwm_topic=general_config['pwm_topic'],
            target_temp_min=general_config['target_temp_min'],
            target_temp_max=general_config['target_temp_max'],
            episode_max_steps=random_config['episode_max_steps'],
            algorithm='Random',
            temp_config=temp_config,
            dist_config=dist_config
        )
        
        with SummaryWriter(log_dir="./tensorboard/Random/") as writer:
            evaluate_env(env, writer, total_timesteps, n_eval_episodes, dist_config) 

    except KeyboardInterrupt:
        print("Program interrupted by user.")

    finally:
        if env is not None:
            env.close()
        print("Environment successfully closed.")

def run_fuzzy(config):
    env = None
    try:
        fuzzy_config = config['fuzzy']
        general_config = config['general']
        temp_config = config['temp_config']
        dist_config = config['dist_config']
        total_timesteps = fuzzy_config['total_timesteps']
        n_eval_episodes = general_config.get('n_eval_episodes', 5)

        env = CustomEnv(
            broker_address=general_config['broker_address'],
            port=general_config['port'],
            temp_topic=general_config['temp_topic'],
            pwm_topic=general_config['pwm_topic'],
            target_temp_min=general_config['target_temp_min'],
            target_temp_max=general_config['target_temp_max'],
            episode_max_steps=fuzzy_config['episode_max_steps'],
            algorithm='Fuzzy',
            temp_config=temp_config,
            dist_config=dist_config
        )
        
        with SummaryWriter(log_dir="./tensorboard/Fuzzy/") as writer:
            evaluate_env(env, writer, total_timesteps, n_eval_episodes, dist_config) 
            
    except KeyboardInterrupt:
        print("Program interrupted by user.")

    finally:
        if env is not None:
            env.close()
        print("Environment successfully closed.")

def run_pid(config):
    env = None
    try:
        pid_config = config['pid']
        general_config = config['general']
        temp_config = config['temp_config']
        dist_config = config['dist_config']
        total_timesteps = pid_config['total_timesteps']
        n_eval_episodes = general_config.get('n_eval_episodes', 5)  # 新增：评估的episode数量

        env = CustomEnv(
            broker_address=general_config['broker_address'],
            port=general_config['port'],
            temp_topic=general_config['temp_topic'],
            pwm_topic=general_config['pwm_topic'],
            target_temp_min=general_config['target_temp_min'],
            target_temp_max=general_config['target_temp_max'],
            episode_max_steps=pid_config['episode_max_steps'],
            algorithm='PID',
            kp=pid_config['kp'],
            ki=pid_config['ki'],
            kd=pid_config['kd'],
            temp_config=temp_config,
            dist_config=dist_config
        )
        
        with SummaryWriter(log_dir="./tensorboard/PID/") as writer:
            evaluate_env(env, writer, total_timesteps, n_eval_episodes, dist_config) 

    except KeyboardInterrupt:
        print("Program interrupted by user.")

    finally:
        if env is not None:
            env.close()
        print("Environment successfully closed.")

def run_sac(config):
    sac_config = config['sac']
    general_config = config['general']
    temp_config = config['temp_config']
    dist_config = config['dist_config']
    n_eval_episodes = general_config.get('n_eval_episodes', 5)
    action_space_type = 'box'  # SAC只支持Box动作空间类型    
    model = None
    try:
        # 创建环境实例
        env = create_sb3_env(general_config, sac_config, 'SAC', action_space_type, temp_config, dist_config, is_eval=False)
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])  # 将环境包装为 Vectorized environment

        # 创建 SAC 模型实例
        # 以下参数均为sb3默认参数
        model = SAC(
            "MlpPolicy", env,
            learning_rate=sac_config.get('learning_rate', 0.0003),
            buffer_size=sac_config.get('buffer_size', 1000000),
            learning_starts=sac_config.get('learning_starts', 100),
            batch_size=sac_config.get('batch_size', 256),
            tau=sac_config.get('tau', 0.005),
            gamma=sac_config.get('gamma', 0.99),
            train_freq=sac_config.get('train_freq', 1),
            gradient_steps=sac_config.get('gradient_steps', 1),
            action_noise=sac_config.get('action_noise', None),
            replay_buffer_class=sac_config.get('replay_buffer_class', None),
            replay_buffer_kwargs=sac_config.get('replay_buffer_kwargs', None),
            optimize_memory_usage=sac_config.get('optimize_memory_usage', False),
            ent_coef=sac_config.get('ent_coef', 'auto'),
            target_update_interval=sac_config.get('target_update_interval', 1),
            target_entropy=sac_config.get('target_entropy', 'auto'),
            use_sde=sac_config.get('use_sde', False),
            sde_sample_freq=sac_config.get('sde_sample_freq', -1),
            use_sde_at_warmup=sac_config.get('use_sde_at_warmup', False),
            stats_window_size=sac_config.get('stats_window_size', 100),
            tensorboard_log=sac_config.get('tensorboard_log', None),
            policy_kwargs=sac_config.get('policy_kwargs', None),
            verbose=sac_config.get('verbose', 2),
            seed=sac_config.get('seed', None),
            device=sac_config.get('device', 'auto')
        )

        # 定义回调
        callbacks = []
        if sac_config['eval_freq'] > 0:
            eval_env = create_sb3_env(general_config, sac_config, 'SAC', action_space_type, temp_config, dist_config, is_eval=True)
            eval_env = Monitor(eval_env)  # 包装评估环境
            eval_env = DummyVecEnv([lambda: eval_env])
            eval_callback = EvalCallback(eval_env, best_model_save_path='./logs_SAC/',
                                         log_path='./logs_SAC/', eval_freq=sac_config['eval_freq'],
                                         deterministic=True, render=False, n_eval_episodes=n_eval_episodes)
            callbacks.append(eval_callback)

        if sac_config['save_freq'] > 0:
            checkpoint_callback = CheckpointCallback(save_freq=sac_config['save_freq'],
                                                     save_path='./checkpoints_SAC/',
                                                     name_prefix='sac_model')
            callbacks.append(checkpoint_callback)

        # 训练模型
        model.learn(total_timesteps=sac_config['total_timesteps'], callback=CallbackList(callbacks), log_interval=1)

    except KeyboardInterrupt:
        print("Program interrupted by user.")

    finally:
        if model:
            model.save("sac_model")
        env.close()
        print("Environment successfully closed.")

def run_ppo(config):
    ppo_config = config['ppo']
    general_config = config['general']
    temp_config = config['temp_config']
    dist_config = config['dist_config']
    n_eval_episodes = general_config.get('n_eval_episodes', 5)
    # PPO支持Box和Discrete动作空间类型，默认连续问题Box
    action_space_type = general_config.get('action_space_type', 'box')     
    model = None
    try:
        # 创建环境实例
        env = create_sb3_env(general_config, ppo_config, 'PPO', action_space_type, temp_config, dist_config, is_eval=False)
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])  # 将环境包装为 Vectorized environment

        # 创建 PPO 模型实例
        # 以下参数均为sb3默认参数
        model = PPO(
            "MlpPolicy", env,
            learning_rate=ppo_config.get('learning_rate', 0.0003),
            n_steps=ppo_config.get('n_steps', 2048),
            batch_size=ppo_config.get('batch_size', 64),
            n_epochs=ppo_config.get('n_epochs', 10),
            gamma=ppo_config.get('gamma', 0.99),
            gae_lambda=ppo_config.get('gae_lambda', 0.95),
            clip_range=ppo_config.get('clip_range', 0.2),
            clip_range_vf=ppo_config.get('clip_range_vf', None),
            normalize_advantage=ppo_config.get('normalize_advantage', True),
            ent_coef=ppo_config.get('ent_coef', 0.0),
            vf_coef=ppo_config.get('vf_coef', 0.5),
            max_grad_norm=ppo_config.get('max_grad_norm', 0.5),
            use_sde=ppo_config.get('use_sde', False),
            sde_sample_freq=ppo_config.get('sde_sample_freq', -1),
            rollout_buffer_class=ppo_config.get('rollout_buffer_class', None),
            rollout_buffer_kwargs=ppo_config.get('rollout_buffer_kwargs', None),
            target_kl=ppo_config.get('target_kl', None),
            stats_window_size=ppo_config.get('stats_window_size', 100),
            tensorboard_log=ppo_config.get('tensorboard_log', None),
            policy_kwargs=ppo_config.get('policy_kwargs', None),
            verbose=ppo_config.get('verbose', 2),
            seed=ppo_config.get('seed', None),
            device=ppo_config.get('device', 'auto')
        )

        # 定义回调
        callbacks = []
        if ppo_config.get('eval_freq', 0) > 0:
            eval_env = create_sb3_env(general_config, ppo_config, 'PPO', action_space_type, temp_config, dist_config, is_eval=True)
            eval_env = Monitor(eval_env)  # 包装评估环境
            eval_env = DummyVecEnv([lambda: eval_env])
            eval_callback = EvalCallback(eval_env, best_model_save_path='./logs_PPO/',
                                         log_path='./logs_PPO/', eval_freq=ppo_config.get('eval_freq', 5000),
                                         deterministic=True, render=False, n_eval_episodes=n_eval_episodes)
            callbacks.append(eval_callback)

        if ppo_config.get('save_freq', 0) > 0:
            checkpoint_callback = CheckpointCallback(save_freq=ppo_config.get('save_freq', 5000),
                                                     save_path='./checkpoints_PPO/',
                                                     name_prefix='ppo_model')
            callbacks.append(checkpoint_callback)

        # 训练模型
        model.learn(total_timesteps=ppo_config.get('total_timesteps', 80000), callback=CallbackList(callbacks), log_interval=1)

    except KeyboardInterrupt:
        print("Program interrupted by user.")

    finally:
        if model:
            model.save("ppo_model")
        env.close()
        print("Environment successfully closed.")

def run_dqn(config):
    dqn_config = config['dqn']
    general_config = config['general']
    temp_config = config['temp_config']
    dist_config = config['dist_config']
    n_eval_episodes = general_config.get('n_eval_episodes', 5)
    action_space_type = 'discrete'  # DQN只支持Discrete动作空间类型
    model = None
    try:
        env = create_sb3_env(general_config, dqn_config, 'DQN', action_space_type, temp_config, dist_config, is_eval=False)
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])

        # 以下参数均为sb3默认参数
        model = DQN(
            "MlpPolicy", env,
            learning_rate=dqn_config.get('learning_rate', 0.0001),
            buffer_size=dqn_config.get('buffer_size', 1000000),
            learning_starts=dqn_config.get('learning_starts', 100),
            batch_size=dqn_config.get('batch_size', 32),
            tau=dqn_config.get('tau', 1.0),
            gamma=dqn_config.get('gamma', 0.99),
            train_freq=dqn_config.get('train_freq', 4),
            gradient_steps=dqn_config.get('gradient_steps', 1),
            replay_buffer_class=dqn_config.get('replay_buffer_class', None),
            replay_buffer_kwargs=dqn_config.get('replay_buffer_kwargs', None),
            optimize_memory_usage=dqn_config.get('optimize_memory_usage', False),
            target_update_interval=dqn_config.get('target_update_interval', 10000),
            exploration_fraction=dqn_config.get('exploration_fraction', 0.1),
            exploration_initial_eps=dqn_config.get('exploration_initial_eps', 1.0),
            exploration_final_eps=dqn_config.get('exploration_final_eps', 0.05),
            max_grad_norm=dqn_config.get('max_grad_norm', 10),
            stats_window_size=dqn_config.get('stats_window_size', 100),
            tensorboard_log=dqn_config.get('tensorboard_log', None),
            policy_kwargs=dqn_config.get('policy_kwargs', None),
            verbose=dqn_config.get('verbose', 2),
            seed=dqn_config.get('seed', None),
            device=dqn_config.get('device', 'auto')
        )

        callbacks = []
        if dqn_config['eval_freq'] > 0:
            eval_env = create_sb3_env(general_config, dqn_config, 'DQN', action_space_type, temp_config, dist_config, is_eval=True)
            eval_env = Monitor(eval_env) 
            eval_env = DummyVecEnv([lambda: eval_env])
            eval_callback = EvalCallback(eval_env, best_model_save_path='./logs_DQN/',
                                         log_path='./logs_DQN/', eval_freq=dqn_config['eval_freq'],
                                         deterministic=True, render=False, n_eval_episodes=n_eval_episodes)
            callbacks.append(eval_callback)

        if dqn_config['save_freq'] > 0:
            checkpoint_callback = CheckpointCallback(save_freq=dqn_config['save_freq'],
                                                     save_path='./checkpoints_DQN/',
                                                     name_prefix='dqn_model')
            callbacks.append(checkpoint_callback)

        model.learn(total_timesteps=dqn_config['total_timesteps'], callback=CallbackList(callbacks), log_interval=1)

    except KeyboardInterrupt:
        print("Program interrupted by user.")

    finally:
        if model:
            model.save("dqn_model")
        env.close()
        print("Environment successfully closed.")

def run_a2c(config):
    a2c_config = config['a2c']
    general_config = config['general']
    temp_config = config['temp_config']
    dist_config = config['dist_config']
    n_eval_episodes = general_config.get('n_eval_episodes', 5)
    # A2C支持Box和Discrete动作空间类型，默认连续问题Box
    action_space_type = general_config.get('action_space_type', 'box')
    model = None
    try:
        # 创建环境实例
        env = create_sb3_env(general_config, a2c_config, 'A2C', action_space_type, temp_config, dist_config, is_eval=False)
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])  # 将环境包装为 Vectorized environment

        # 创建 A2C 模型实例
        model = A2C(
            "MlpPolicy", env,
            learning_rate=a2c_config.get('learning_rate', 0.0007),
            n_steps=a2c_config.get('n_steps', 5),
            gamma=a2c_config.get('gamma', 0.99),
            gae_lambda=a2c_config.get('gae_lambda', 1.0),
            ent_coef=a2c_config.get('ent_coef', 0.0),
            vf_coef=a2c_config.get('vf_coef', 0.5),
            max_grad_norm=a2c_config.get('max_grad_norm', 0.5),
            rms_prop_eps=a2c_config.get('rms_prop_eps', 1e-05),
            use_rms_prop=a2c_config.get('use_rms_prop', True),
            use_sde=a2c_config.get('use_sde', False),
            sde_sample_freq=a2c_config.get('sde_sample_freq', -1),
            rollout_buffer_class=a2c_config.get('rollout_buffer_class', None),
            rollout_buffer_kwargs=a2c_config.get('rollout_buffer_kwargs', None),
            normalize_advantage=a2c_config.get('normalize_advantage', False),
            stats_window_size=a2c_config.get('stats_window_size', 100),
            tensorboard_log=a2c_config.get('tensorboard_log', None),
            policy_kwargs=a2c_config.get('policy_kwargs', None),
            verbose=a2c_config.get('verbose', 2),
            seed=a2c_config.get('seed', None),
            device=a2c_config.get('device', 'auto')
        )

        # 定义回调
        callbacks = []
        if a2c_config.get('eval_freq', 0) > 0:
            eval_env = create_sb3_env(general_config, a2c_config, 'A2C', action_space_type, temp_config, dist_config, is_eval=True)
            eval_env = Monitor(eval_env)  # 包装评估环境
            eval_env = DummyVecEnv([lambda: eval_env])
            eval_callback = EvalCallback(eval_env, best_model_save_path='./logs_A2C/',
                                         log_path='./logs_A2C/', eval_freq=a2c_config.get('eval_freq', 5000),
                                         deterministic=True, render=False, n_eval_episodes=n_eval_episodes)
            callbacks.append(eval_callback)

        if a2c_config.get('save_freq', 0) > 0:
            checkpoint_callback = CheckpointCallback(save_freq=a2c_config.get('save_freq', 5000),
                                                     save_path='./checkpoints_A2C/',
                                                     name_prefix='a2c_model')
            callbacks.append(checkpoint_callback)

        # 训练模型
        model.learn(total_timesteps=a2c_config.get('total_timesteps', 80000), callback=CallbackList(callbacks), log_interval=1)

    except KeyboardInterrupt:
        print("Program interrupted by user.")

    finally:
        if model:
            model.save("a2c_model")
        env.close()
        print("Environment successfully closed.")

def run_ddpg(config):
    ddpg_config = config['ddpg']
    general_config = config['general']
    temp_config = config['temp_config']
    dist_config = config['dist_config']
    n_eval_episodes = general_config.get('n_eval_episodes', 5)
    action_space_type = 'box'  # DDPG只支持Box动作空间类型    
    model = None
    try:
        # 创建环境实例
        env = create_sb3_env(general_config, ddpg_config, 'DDPG', action_space_type, temp_config, dist_config, is_eval=False)
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])  # 将环境包装为 Vectorized environment

        # 创建 DDPG 模型实例
        model = DDPG(
            "MlpPolicy", env,
            learning_rate=ddpg_config.get('learning_rate', 0.001),
            buffer_size=ddpg_config.get('buffer_size', 1000000),
            learning_starts=ddpg_config.get('learning_starts', 100),
            batch_size=ddpg_config.get('batch_size', 256),
            tau=ddpg_config.get('tau', 0.005),
            gamma=ddpg_config.get('gamma', 0.99),
            train_freq=ddpg_config.get('train_freq', 1),
            gradient_steps=ddpg_config.get('gradient_steps', 1),
            action_noise=ddpg_config.get('action_noise', None),
            replay_buffer_class=ddpg_config.get('replay_buffer_class', None),
            replay_buffer_kwargs=ddpg_config.get('replay_buffer_kwargs', None),
            optimize_memory_usage=ddpg_config.get('optimize_memory_usage', False),
            tensorboard_log=ddpg_config.get('tensorboard_log', None),
            policy_kwargs=ddpg_config.get('policy_kwargs', None),
            verbose=ddpg_config.get('verbose', 2),
            seed=ddpg_config.get('seed', None),
            device=ddpg_config.get('device', 'auto')
        )

        # 定义回调
        callbacks = []
        if ddpg_config.get('eval_freq', 0) > 0:
            eval_env = create_sb3_env(general_config, ddpg_config, 'DDPG', action_space_type, temp_config, dist_config, is_eval=True)
            eval_env = Monitor(eval_env)  # 包装评估环境
            eval_env = DummyVecEnv([lambda: eval_env])
            eval_callback = EvalCallback(eval_env, best_model_save_path='./logs_DDPG/',
                                         log_path='./logs_DDPG/', eval_freq=ddpg_config.get('eval_freq', 5000),
                                         deterministic=True, render=False, n_eval_episodes=n_eval_episodes)
            callbacks.append(eval_callback)

        if ddpg_config.get('save_freq', 0) > 0:
            checkpoint_callback = CheckpointCallback(save_freq=ddpg_config.get('save_freq', 5000),
                                                     save_path='./checkpoints_DDPG/',
                                                     name_prefix='ddpg_model')
            callbacks.append(checkpoint_callback)

        # 训练模型
        model.learn(total_timesteps=ddpg_config.get('total_timesteps', 80000), callback=CallbackList(callbacks), log_interval=1)

    except KeyboardInterrupt:
        print("Program interrupted by user.")

    finally:
        if model:
            model.save("ddpg_model")
        env.close()
        print("Environment successfully closed.")

def run_td3(config):
    td3_config = config['td3']
    general_config = config['general']
    temp_config = config['temp_config']
    dist_config = config['dist_config']
    n_eval_episodes = general_config.get('n_eval_episodes', 5)
    action_space_type = 'box'  # TD3只支持Box动作空间类型    
    model = None
    try:
        # 创建环境实例
        env = create_sb3_env(general_config, td3_config, 'TD3', action_space_type, temp_config, dist_config, is_eval=False)
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])  # 将环境包装为 Vectorized environment

        # 创建 TD3 模型实例
        model = TD3(
            "MlpPolicy", env,
            learning_rate=td3_config.get('learning_rate', 0.001),
            buffer_size=td3_config.get('buffer_size', 1000000),
            learning_starts=td3_config.get('learning_starts', 100),
            batch_size=td3_config.get('batch_size', 256),
            tau=td3_config.get('tau', 0.005),
            gamma=td3_config.get('gamma', 0.99),
            train_freq=td3_config.get('train_freq', 1),
            gradient_steps=td3_config.get('gradient_steps', 1),
            action_noise=td3_config.get('action_noise', None),
            replay_buffer_class=td3_config.get('replay_buffer_class', None),
            replay_buffer_kwargs=td3_config.get('replay_buffer_kwargs', None),
            optimize_memory_usage=td3_config.get('optimize_memory_usage', False),
            policy_delay=td3_config.get('policy_delay', 2),
            target_policy_noise=td3_config.get('target_policy_noise', 0.2),
            target_noise_clip=td3_config.get('target_noise_clip', 0.5),
            stats_window_size=td3_config.get('stats_window_size', 100),
            tensorboard_log=td3_config.get('tensorboard_log', None),
            policy_kwargs=td3_config.get('policy_kwargs', None),
            verbose=td3_config.get('verbose', 2),
            seed=td3_config.get('seed', None),
            device=td3_config.get('device', 'auto')
        )

        # 定义回调
        callbacks = []
        if td3_config['eval_freq'] > 0:
            eval_env = create_sb3_env(general_config, td3_config, 'TD3', action_space_type, temp_config, dist_config, is_eval=True)
            eval_env = Monitor(eval_env)  # 包装评估环境
            eval_env = DummyVecEnv([lambda: eval_env])
            eval_callback = EvalCallback(eval_env, best_model_save_path='./logs_TD3/',
                                         log_path='./logs_TD3/', eval_freq=td3_config['eval_freq'],
                                         deterministic=True, render=False, n_eval_episodes=n_eval_episodes)
            callbacks.append(eval_callback)

        if td3_config['save_freq'] > 0:
            checkpoint_callback = CheckpointCallback(save_freq=td3_config['save_freq'],
                                                     save_path='./checkpoints_TD3/',
                                                     name_prefix='td3_model')
            callbacks.append(checkpoint_callback)

        # 训练模型
        model.learn(total_timesteps=td3_config['total_timesteps'], callback=CallbackList(callbacks), log_interval=1)

    except KeyboardInterrupt:
        print("Program interrupted by user.")

    finally:
        if model:
            model.save("td3_model")
        env.close()
        print("Environment successfully closed.")

def run_trpo(config):
    trpo_config = config['trpo']
    general_config = config['general']
    temp_config = config['temp_config']
    dist_config = config['dist_config']
    n_eval_episodes = general_config.get('n_eval_episodes', 5)
    action_space_type = general_config.get('action_space_type', 'box')
    model = None
    try:
        # 创建环境实例
        env = create_sb3_env(general_config, trpo_config, 'TRPO', action_space_type, temp_config, dist_config, is_eval=False)
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])  # 将环境包装为 Vectorized environment

        # 创建 TRPO 模型实例
        model = TRPO(
            "MlpPolicy", env,
            learning_rate=trpo_config.get('learning_rate', 0.001),
            n_steps=trpo_config.get('n_steps', 2048),
            batch_size=trpo_config.get('batch_size', 128),
            gamma=trpo_config.get('gamma', 0.99),
            cg_max_steps=trpo_config.get('cg_max_steps', 15),
            cg_damping=trpo_config.get('cg_damping', 0.1),
            line_search_shrinking_factor=trpo_config.get('line_search_shrinking_factor', 0.8),
            line_search_max_iter=trpo_config.get('line_search_max_iter', 10),
            n_critic_updates=trpo_config.get('n_critic_updates', 10),
            gae_lambda=trpo_config.get('gae_lambda', 0.95),
            use_sde=trpo_config.get('use_sde', False),
            sde_sample_freq=trpo_config.get('sde_sample_freq', -1),
            rollout_buffer_class=trpo_config.get('rollout_buffer_class', None),
            rollout_buffer_kwargs=trpo_config.get('rollout_buffer_kwargs', None),
            normalize_advantage=trpo_config.get('normalize_advantage', True),
            target_kl=trpo_config.get('target_kl', 0.01),
            sub_sampling_factor=trpo_config.get('sub_sampling_factor', 1),
            stats_window_size=trpo_config.get('stats_window_size', 100),
            tensorboard_log=trpo_config.get('tensorboard_log', None),
            policy_kwargs=trpo_config.get('policy_kwargs', None),
            verbose=trpo_config.get('verbose', 2),
            seed=trpo_config.get('seed', None),
            device=trpo_config.get('device', 'auto')
        )

        # 定义回调
        callbacks = []
        if trpo_config.get('eval_freq', 0) > 0:
            eval_env = create_sb3_env(general_config, trpo_config, 'TRPO', action_space_type, temp_config, dist_config, is_eval=True)
            eval_env = Monitor(eval_env)  # 包装评估环境
            eval_env = DummyVecEnv([lambda: eval_env])
            eval_callback = EvalCallback(eval_env, best_model_save_path='./logs_TRPO/',
                                         log_path='./logs_TRPO/', eval_freq=trpo_config.get('eval_freq', 5000),
                                         deterministic=True, render=False, n_eval_episodes=n_eval_episodes)
            callbacks.append(eval_callback)

        if trpo_config.get('save_freq', 0) > 0:
            checkpoint_callback = CheckpointCallback(save_freq=trpo_config.get('save_freq', 5000),
                                                     save_path='./checkpoints_TRPO/',
                                                     name_prefix='trpo_model')
            callbacks.append(checkpoint_callback)

        # 训练模型
        model.learn(total_timesteps=trpo_config.get('total_timesteps', 80000), callback=CallbackList(callbacks), log_interval=1)

    except KeyboardInterrupt:
        print("Program interrupted by user.")

    finally:
        if model:
            model.save("trpo_model")
        env.close()
        print("Environment successfully closed.")        

def main():
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    algorithm = config['general']['algorithm']
    action_space_type = config['general'].get('action_space_type', 'box')
    
    discrete_step = config['general'].get('discrete_step', 10)  # 新增读取离散步长
    
    dist_config = config.get('dist_config', None)

    if 200 % discrete_step != 0:  # 验证总范围200%是否能被步长整除
        raise ValueError(f"discrete_step {discrete_step} is invalid. 200% range must be divisible by the step size.")    

    if algorithm == 'PID':
        run_pid(config)
    elif algorithm == 'Fuzzy':
        run_fuzzy(config)
    elif algorithm == 'Random':
        run_random(config)
    elif algorithm == 'SAC':
        if action_space_type != 'box':
            print("Warning: SAC only supports 'box' action_space_type. Ignoring and continuing.")
        run_sac(config)
    elif algorithm == 'PPO':
        if action_space_type not in ['box', 'discrete']:
            print("Warning: PPO supports 'box' or 'discrete' action_space_type. Invalid input. Defaulting to 'box'.")
            action_space_type = 'box'
        run_ppo(config)
    elif algorithm == 'DQN':
        if action_space_type != 'discrete':
            print("Warning: DQN only supports 'discrete' action_space_type. Ignoring and continuing.")
        run_dqn(config)
    elif algorithm == 'A2C':
        if action_space_type not in ['box', 'discrete']:
            print("Warning: A2C supports 'box' or 'discrete' action_space_type. Invalid input. Defaulting to 'box'.")
            action_space_type = 'box'
        run_a2c(config)
    elif algorithm == 'DDPG':  
        if action_space_type != 'box':
            print("Warning: DDPG only supports 'box' action_space_type. Ignoring and continuing.")
        run_ddpg(config)
    elif algorithm == 'TD3':  
        if action_space_type != 'box':
            print("Warning: TD3 only supports 'box' action_space_type. Ignoring and continuing.")
        run_td3(config)
    elif algorithm == 'TRPO':  
        if action_space_type not in ['box', 'discrete']:
            print("Warning: TRPO supports 'box' or 'discrete' action_space_type. Invalid input. Defaulting to 'box'.")
            action_space_type = 'box'
        run_trpo(config)
    else:
        raise ValueError("Unsupported algorithm specified in config file.")

if __name__ == "__main__":
    main()

# Environment and Model Training Setup

## Overview

This repository provides the implementation of various algorithms for temperature control, including Random, On-Off, Fuzzy, PID, and DRL algorithms (SAC, PPO, DQN, A2C, DDPG, TD3, TRPO). The environment simulates a temperature control system with the option to introduce disturbances.

## Requirements

### PC Environment

```plaintext
python==3.8.19
gym==0.26.2
gymnasium==0.29.1
matplotlib==3.7.5
numpy==1.24.4
paho-mqtt==1.6.1
pandas==2.0.3
pyyaml==6.0.1
sb3-contrib==2.3.0
scikit-fuzzy==0.4.2
stable-baselines3==2.3.1
tensorboard==2.14.0
tensorboard-data-server==0.7.2
torch==2.2.2
```

### Raspberry Pi Environment

```plaintext
python==3.9.2
daqhats==1.4.0.8
numpy==1.24.3
paho-mqtt==1.6.1
RPi.GPIO==0.7.0
rpi-hardware-pwm==0.1.4
```

**Note**: The `daqhats` package can be found [here](https://github.com/mccdaq/daqhats/releases).

## Usage

### Running the Model

1. **Train the Model**:
   To train a model, configure the `config.yaml` file with the desired algorithm and parameters. Then, run the `train.py` script:

   ```bash
   python train.py
   ```

2. **Evaluate the Model**:
   After training, evaluate the model by running the `test_model.py` script:

   ```bash
   python test_model.py --model_path path_to_your_model.zip --algorithm <Algorithm_Name>
   ```

   Replace `<Algorithm_Name>` with the name of the algorithm you wish to evaluate.

   Example:

   ```bash
   python test_model.py --model_path C:\Users\AAA\Desktop\torch_sb3\a2c_model_box.zip --algorithm A2C --action_space_type box --episode_max_steps 500 --n_eval_episodes 50
   ```

### Setting Up MQTT on Raspberry Pi

When setting up the MQTT broker on the Raspberry Pi, ensure that the server is not restricted to only `localhost`. This allows for proper communication between the Raspberry Pi and other devices on the network.

## Configuration

### `config.yaml`

This file contains the configuration settings for training and evaluating models. You can specify the algorithm, action space type, number of episodes, temperature control settings, and more.

## Logging

- **TensorBoard**: All training and evaluation metrics are logged to TensorBoard. You can view the logs using the following command:

  ```bash
  tensorboard --logdir=./tensorboard/ --bind_all
  ```

## License

This project is licensed under the MIT License.

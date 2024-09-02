# mqtt_pi.py
import paho.mqtt.client as mqtt
from daqhats import mcc134, HatIDs, TcTypes
from daqhats_utils import select_hat_device
from time import sleep
import RPi.GPIO as GPIO
import csv
import time
import datetime
import numpy as np
import json  
from rpi_hardware_pwm import HardwarePWM

MQTT_BROKER = "127.0.0.1"  # IP Address of MQTT Server # In the meantime, the server runs on Pi
MQTT_PORT = 1883
MQTT_TOPIC_TEMP = "temperature_control/temp_data"
MQTT_TOPIC_PWM = "temperature_control/pwm_control"


MAX_RESTARTS = 5
RESTART_DELAY = 5  

restart_attempts = 0

while restart_attempts < MAX_RESTARTS:
    try:
        address = select_hat_device(HatIDs.MCC_134)
        hat = mcc134(address)
        channels = (0, 1, 2, 3)
        for channel in channels:
            hat.tc_type_write(channel, TcTypes.TYPE_K)

        pwm_heat = HardwarePWM(pwm_channel=0, hz=100)  # Heating PWM
        pwm_cool = HardwarePWM(pwm_channel=1, hz=2000)  # Cooling PWM
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(16, GPIO.OUT)
        pwm_dist = GPIO.PWM(16, 100)  # Disturbance PWM
        pwm_heat.stop()
        pwm_cool.stop()
        pwm_dist.stop()
        pwm_heat.start(0)
        pwm_cool.start(0)
        pwm_dist.start(0)

        # get temp
        def Temperature(channel):
            temp = hat.t_in_read(channel)
            if temp not in [mcc134.OPEN_TC_VALUE, mcc134.OVERRANGE_TC_VALUE, mcc134.COMMON_MODE_TC_VALUE]:
                return temp
            else:
                return None

        def on_connect(client, userdata, flags, rc):
            print("Connected with result code "+str(rc))
            client.subscribe(MQTT_TOPIC_PWM)

        def on_message(client, userdata, msg):
            message = str(msg.payload.decode("utf-8"))
            # print(f"Received message on {msg.topic}: {message}")         
            values = message.split(',')
            # print(f"Parsed values: {values}")  # 添加的打印语句
            
            if len(values) == 3:
                heat_dc, cool_dc, dist_dc = map(float, values)
            elif len(values) == 2:
                heat_dc, cool_dc = map(float, values)
                dist_dc = 0  # 默认为0，或根据需要设置其他默认值
            else:
                # 处理异常情况，例如日志记录
                print(f"Received an invalid message: {message}")
                return            
            
            pwm_heat.change_duty_cycle(heat_dc)
            pwm_cool.change_duty_cycle(cool_dc)
            pwm_dist.ChangeDutyCycle(dist_dc)  # 添加dist信号的处理

            # Get current temperatures
            actual_temp = sum(Temperature(channel) for channel in channels[:2] if Temperature(channel) is not None) / 2
            cooling_temp = sum(Temperature(channel) for channel in channels[2:] if Temperature(channel) is not None) / 2

            # Print temperatures and PWM signals
            print(f"Temperature: Actual: {actual_temp:.3f}, Cooling: {cooling_temp:.3f} | PWM: Heat DC: {heat_dc:06.3f}, Cool DC: {cool_dc:06.3f}, Dist DC: {dist_dc:06.3f}")

        # set up MQTT client
        client = mqtt.Client()
        client.on_connect = on_connect
        client.on_message = on_message

        client.connect(MQTT_BROKER, MQTT_PORT, 120)
        client.loop_start()
            
        last_time = time.time() 

        while True:
            current_time = time.time()
            if current_time - last_time >= 0.1:
                
                actual_temp = sum(Temperature(channel) for channel in channels[:2] if Temperature(channel) is not None) / 2
                cooling_temp = sum(Temperature(channel) for channel in channels[2:] if Temperature(channel) is not None) / 2
                
                # creat json pack with 2 temp data
                temp_data = json.dumps({"actual_temp": actual_temp, "cooling_temp": cooling_temp})
                client.publish(MQTT_TOPIC_TEMP, temp_data)

    except Exception as e:
        print(f"Program encountered an error and will restart in {RESTART_DELAY} seconds: {e}")
        restart_attempts += 1
        pwm_heat.stop()
        pwm_cool.stop()
        pwm_dist.stop()
        GPIO.cleanup()
        client.disconnect()
        client.loop_stop()
        time.sleep(RESTART_DELAY)

    except KeyboardInterrupt:
        print("Program manually interrupted.")
        pwm_heat.stop()
        pwm_cool.stop()
        pwm_dist.stop()
        GPIO.cleanup()
        client.disconnect()
        client.loop_stop()
        break  

if restart_attempts >= MAX_RESTARTS:
    print(f"Program failed to restart {MAX_RESTARTS} times and will now stop.")


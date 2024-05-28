#mqtt_win.py

import paho.mqtt.client as mqtt  # 导入MQTT客户端库  # 注意：使用1.6.1的paho-mqtt
import json  
import time  

class MQTTClient:
    def __init__(self, broker_address="100.120.27.64", port=1883, temp_topic="temperature_control/temp_data", pwm_topic="temperature_control/pwm_control"):
        # 类初始化方法，设置MQTT服务器地址、端口、订阅主题和发布主题，默认值已给出
        self.broker_address = broker_address  # MQTT代理服务器地址
        self.port = port  # MQTT服务器端口
        self.temp_topic = temp_topic  # 订阅的温度主题
        self.pwm_topic = pwm_topic  # 发布PWM信号的主题
        self.client = mqtt.Client()  # 创建MQTT客户端实例
        self.client.on_connect = self.on_connect  # 设置连接回调函数
        self.client.on_message = self.on_message  # 设置接收消息回调函数
        self.actual_temp = 0  # 初始化实际温度值
        self.cooling_temp = 0  # 初始化冷却温度值

    def on_connect(self, client, userdata, flags, rc):
        print(f"Connected with result code {rc}")
        if rc == 0:
            self.client.subscribe(self.temp_topic)
            print(f"Subscribed to topic {self.temp_topic}")
        else:
            print(f"Failed to connect, return code {rc}")

    def on_message(self, client, userdata, msg):
        try:
            temp_data = json.loads(msg.payload.decode('utf-8'))
            new_actual_temp = temp_data.get("actual_temp", self.actual_temp)
            new_cooling_temp = temp_data.get("cooling_temp", self.cooling_temp)
            if new_actual_temp != self.actual_temp or new_cooling_temp != self.cooling_temp:
                self.actual_temp = new_actual_temp
                self.cooling_temp = new_cooling_temp
                # print(f"Updated temperatures - actual_temp: {self.actual_temp}, cooling_temp: {self.cooling_temp}")
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
        except Exception as e:
            print("Unexpected error in on_message:", e)

    def connect(self):
        try:
            self.client.connect(self.broker_address, self.port, 60)
            self.client.loop_start()
            print(f"Connecting to MQTT broker at {self.broker_address}:{self.port}")
        except Exception as e:
            print("Error connecting to MQTT broker:", e)

    def disconnect(self):
        try:
            self.client.loop_stop()  # 停止网络循环
            print("Sending stop PWM signal...")
            for _ in range(3):
                self.client.publish(self.pwm_topic, "0,0,0")  # 尝试发送停止PWM信号
                time.sleep(0.5)
            time.sleep(2)
            self.client.disconnect()  # 尝试断开与MQTT服务器的连接
            print("Disconnected from MQTT broker.")
        except Exception as e:
            print("Error during disconnection: ", e)
            # 这里可以添加更多的错误处理逻辑

    def publish_pwm(self, heat_dc, cool_dc, dist_dc=0):
        try:
            self.client.publish(self.pwm_topic, f"{heat_dc},{cool_dc},{dist_dc}")  # 尝试向指定主题发布消息
        except Exception as e:
            print("Error publishing PWM signal: ", e)
            # 这里可以添加更多的错误处理逻辑

    def get_temperature(self):
        if self.actual_temp == 0 and self.cooling_temp == 0:
            print("Temperature data is not yet available or still zero.")
        return self.actual_temp, self.cooling_temp
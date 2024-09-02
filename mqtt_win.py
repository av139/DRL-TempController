#mqtt_win.py

import paho.mqtt.client as mqtt  # Import the MQTT client library (Note: Use paho-mqtt version 1.6.1)
import json  
import time  

class MQTTClient:
    def __init__(self, broker_address="100.120.27.64", port=1883, temp_topic="temperature_control/temp_data", pwm_topic="temperature_control/pwm_control"):
        # Initialization method for the class, setting MQTT broker address, port, subscription topic, and publishing topic, with default values provided
        self.broker_address = broker_address  # MQTT broker server address
        self.port = port  # MQTT broker port
        self.temp_topic = temp_topic  # Topic for subscribing to temperature data
        self.pwm_topic = pwm_topic  # Topic for publishing PWM signals
        self.client = mqtt.Client()  # Create an instance of the MQTT client
        self.client.on_connect = self.on_connect  # Set the connection callback function
        self.client.on_message = self.on_message  # Set the message reception callback function
        self.actual_temp = 0  # Initialize the actual temperature value
        self.cooling_temp = 0  # Initialize the cooling temperature value

    # When connected to the MQTT broker
    def on_connect(self, client, userdata, flags, rc):
        print(f"Connected with result code {rc}")
        if rc == 0:
            self.client.subscribe(self.temp_topic)
            print(f"Subscribed to topic {self.temp_topic}")
        else:
            print(f"Failed to connect, return code {rc}")

    # When the temperature topic is updated
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

    # Establish a connection to the MQTT broker
    def connect(self):
        try:
            self.client.connect(self.broker_address, self.port, 60)
            self.client.loop_start()
            print(f"Connecting to MQTT broker at {self.broker_address}:{self.port}")
        except Exception as e:
            print("Error connecting to MQTT broker:", e)

    # Disconnect from the MQTT broker
    def disconnect(self):
        try:
            self.client.loop_stop()  # Stop the network loop
            print("Sending stop PWM signal...")
            for _ in range(3):
                self.client.publish(self.pwm_topic, "0,0,0")  # Attempt to send PWM stop signal multiple times
                time.sleep(0.5) 
            time.sleep(2)  # Ensure the action is executed
            self.client.disconnect()  # Attempt to disconnect from the MQTT broker
            print("Disconnected from MQTT broker.")
        except Exception as e:
            print("Error during disconnection: ", e)
            # Additional error handling logic can be added here

    # Publish PWM signals
    def publish_pwm(self, heat_dc, cool_dc, dist_dc=0):
        try:
            self.client.publish(self.pwm_topic, f"{heat_dc},{cool_dc},{dist_dc}")  # Attempt to publish a message to the specified topic
        except Exception as e:
            print("Error publishing PWM signal: ", e)
            # Additional error handling logic can be added here

    # Retrieve temperature data
    def get_temperature(self):
        if self.actual_temp == 0 and self.cooling_temp == 0:
            print("Temperature data is not yet available or still zero.")
        return self.actual_temp, self.cooling_temp

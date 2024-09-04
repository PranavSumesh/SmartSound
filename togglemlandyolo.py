import cv2
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.clock import Clock
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
import pandas as pd
from ultralytics import YOLO
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from datetime import datetime
from tracker import *

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import datetime
# Code 2: People Detection
model = YOLO('yolov8s.pt')
# State variables
# Define state variables
ML_MODEL = "ml_model"
PERSON_DETECTION = "person_detection"
current_state = PERSON_DETECTION
pause_detection = False

# Code 1: Volume Control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))


class VolumeControlApp(App):
    def build(self):
        layout = BoxLayout(orientation='vertical')

        self.volume_label = Label(text='Volume: ')
        self.volume_slider = Slider(min=-63, max=0, value=volume.GetMasterVolumeLevel(), step=1)
        self.volume_slider.bind(value=self.set_volume)

        self.inc_volume_button = Button(text='Increase Volume')
        self.inc_volume_button.bind(on_press=self.increase_volume)

        self.dec_volume_button = Button(text='Decrease Volume')
        self.dec_volume_button.bind(on_press=self.decrease_volume)

        self.toggle_detection_button = ToggleButton(text='Pause Detection')
        self.toggle_detection_button.bind(on_press=self.toggle_detection)

        self.resume_detection_button = Button(text='Resume Detection')
        self.resume_detection_button.bind(on_press=self.resume_detection)

        self.toggle_mode_button = Button(text='Toggle Mode')
        self.toggle_mode_button.bind(on_press=self.toggle_mode)

        self.video_label = Image()

        layout.add_widget(self.volume_label)
        layout.add_widget(self.volume_slider)
        layout.add_widget(self.inc_volume_button)
        layout.add_widget(self.dec_volume_button)
        layout.add_widget(self.toggle_detection_button)
        layout.add_widget(self.resume_detection_button)
        layout.add_widget(self.toggle_mode_button)
        layout.add_widget(self.video_label)

        Clock.schedule_interval(self.update_ui, 1.0 / 60.0)  # Update UI every frame
        Clock.schedule_interval(self.update_video_display, 1.0 / 60.0)  # Update video display every frame

        return layout

    def resume_detection(self, instance):
        global pause_detection
        pause_detection = False

    def set_volume(self, instance, value):
        volume.SetMasterVolumeLevel(value, None)
        self.volume_label.text = f'Volume: {value} dB'

    def increase_volume(self, instance):
        current_volume = volume.GetMasterVolumeLevel()
        self.set_volume(None, current_volume + 1)

    def decrease_volume(self, instance):
        current_volume = volume.GetMasterVolumeLevel()
        self.set_volume(None, current_volume - 1)

    def toggle_detection(self, instance):
        global pause_detection
        pause_detection = not pause_detection

    def toggle_mode(self, instance):
        global current_state
        if current_state == PERSON_DETECTION:
            current_state = ML_MODEL
        else:
            current_state = PERSON_DETECTION

    def update_ui(self, dt):
        self.volume_slider.value = volume.GetMasterVolumeLevel()

    def update_video_display(self, dt):
        if not pause_detection:
            ret, frame = cap.read()
            current_volume_db = volume.GetMasterVolumeLevel()

            if current_state == PERSON_DETECTION:
                results = model.predict(frame)
                a = results[0].boxes.data
                px = pd.DataFrame(a).astype("float")
                list = []
                countperson = 0
                cx1 = frame.shape[1] // 2   # Horizontal center of the screen
                
                left_of_green_line = 0  # Initialize variables to count people to the left and right of the green line
                right_of_green_line = 0

                for index, row in px.iterrows():
                    x1 = int(row[0])
                    y1 = int(row[1])
                    x2 = int(row[2])
                    y2 = int(row[3])
                    d = int(row[5])

                    c = class_list[d]
                    if 'person' in c:
                        countperson += 1
                        list.append([x1, y1, x2, y2])
                        center_x = (x1 + x2) // 2
                        if center_x < cx1:
                            left_of_green_line += 1
                        elif center_x > cx1:
                            right_of_green_line += 1

                cv2.putText(frame, f'Left : {left_of_green_line}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f'Right : {right_of_green_line}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Rest of the code for volume control based on person detection
                if countperson >= 0:
                    print('Volume:', current_volume_db)

                    if current_volume_db < -63:
                        print('Minimum volume')
                    elif current_volume_db > 0:
                        print('Maximum volume')
                    else:
                        if countperson == 1:
                            # Set volume to 20%
                            volume.SetMasterVolumeLevel(-27.0, None)
                            volume.SetMute(0, None)
                            print('Volume set to 20%')
                        elif countperson == 2:
                            # Set volume to 30%
                            volume.SetMasterVolumeLevel(-18.0, None)
                            volume.SetMute(0, None)
                            print('Volume set to 30%')
                        elif countperson == 3:
                            # Set volume to 30%
                            volume.SetMasterVolumeLevel(-18.0, None)
                            volume.SetMute(0, None)
                            print('Volume set to 30%')
                        elif countperson == 4:
                            # Set volume to 40%
                            volume.SetMasterVolumeLevel(-9.0, None)
                            volume.SetMute(0, None)
                            print('Volume set to 40%')
                        elif countperson >= 5:
                            # Set volume to 100%
                            volume.SetMasterVolumeLevel(0.0, None)
                            volume.SetMute(0, None)
                            print('Volume set to 100%')
                        elif countperson == 0:
                            # Mute the speaker
                            volume.SetMasterVolumeLevel(-63.0, None)
                            volume.SetMute(1, None)
                            print('Speaker muted')
                        
                        # Get the current date, time, day of the week, and volume
                        current_time = datetime.datetime.now().strftime("%H:%M:%S")
                        current_day_of_week = datetime.datetime.now().strftime("%A")
                        
                        current_volume_percent = volume.GetMasterVolumeLevel()
                        

                        # Writing date, time, day of the week, and volume to CSV file
                        with open('data.csv', 'a') as f:
                            f.write(f'{current_time},{current_day_of_week},{current_volume_percent}\n')


                        bbox_id = tracker.update(list)

                        for bbox in bbox_id:
                                x3, y3, x4, y4, id = bbox
                                cx = (x3 + x4) // 2
                                cy = (y3 + y4) // 2
                                cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)

                            # Draw the vertical green line
                        cx1 = frame.shape[1] // 2  # Horizontal center of the screen
                        cv2.line(frame, (cx1, 5), (cx1, 495), (0, 255, 0), 2)

                        cv2.imshow("RGB", frame)

           


            elif current_state == ML_MODEL:
                # Load the dataset from CSV
                df = pd.read_csv("data.csv")

                # Preprocessing
                df["Time"] = df["Time"].apply(lambda x: int(x.split(".")[0].split(":")[0]))  # Extract hour from time
                df["Day"] = df["Day"].replace({"Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4, "Friday": 5, "Saturday": 6, "Sunday": 7})  # Map days of the week to numerical values

                # Split features and target variable
                X = df[["Time", "Day"]]
                y = df["Volume"]

                # Split data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train the decision tree regression model with hyperparameter tuning
                modell = DecisionTreeRegressor(max_depth=5, min_samples_split=5, min_samples_leaf=2)
                modell.fit(X_train, y_train)

                # Predict on the test set
                y_pred = modell.predict(X_test)

                # Evaluate the model
                mse = mean_squared_error(y_test, y_pred)
                print("Mean Squared Error:", mse)


                # Get current time and day
                current_time = datetime.datetime.now().hour
                current_day = datetime.datetime.now().weekday()  # Monday is 0 and Sunday is 6

                predicted_volume = modell.predict([[current_time, current_day]])

                print("Predicted volume for:", predicted_volume)
                volume.SetMute(0, None)

                volume.SetMasterVolumeLevel(predicted_volume, None)
                self.volume_label.text = f'Volume: {predicted_volume} dB'

            cv2.imshow("RGB", frame)

    def on_stop(self):
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    my_file = open("coco.txt", "r")
    data = my_file.read()
    class_list = data.split("\n")
    count = 0
    tracker = Tracker()
    counter1 = {}
    persondown = {}
    personup = {}
    counter2 = {}

    VolumeControlApp().run()

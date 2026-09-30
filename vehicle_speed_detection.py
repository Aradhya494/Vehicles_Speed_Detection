import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import ImageTk, Image
from custom_sort import Sort
from threading import Thread
import cv2
import time
import numpy as np
from ultralytics import YOLO
import sys


# calibration for real-world measurements
CALIBRATION_DISTANCE = 5 # in meters
CALIBRATION_PIXELS = 200 # in pixels
PIXEL_TO_METER = CALIBRATION_DISTANCE / CALIBRATION_PIXELS
FRAME_RATE = 30

#Global variables
stop_tracking = False
video_path = "TrafficVideo.mp4"
tracker = Sort()
model = YOLO('yolov8n.pt')
log_window = None

#load class names
classnames = model.names
with open('classes.txt' , 'r') as f:
    classnames = f.read().splitlines()


def open_log_window():
    """open a separate window for logs."""
    global log_window, log_text_widget
    if log_window is None or not tk.Toplevel.winfo_exists(log_window):
       log_window = tk.Toplevel()
       log_window.title("log window")
       log_window.geometry("600x400")

       log_text_widget = tk.Text(log_window, wrap="word" , font=("Helvetica" , 10))
       log_text_widget.pack(expand=True, fill=tk.BOTH)


def log_message(message):
    try:
        log_text_widget.insert(tk.END, message + "\n")
        log_text_widget.see(tk.END)
    except NameError:
        print(message)

def _update_log_widget(message):
    """Update the log widget."""
    log_text_widget.insert(tk.END, message + "\n")
    log_text_widget.see(tk.END)


def start_tracking(canva, info_label):
    global stop_tracking, video_path, tracker

    if not video_path:
        messagebox.showerror("Error", "please select a videofile first!")
        return

    stop_tracking = False
    tracker = Sort(max_age=30)
    cap = cv2.VideoCapture(video_path)
    prev_positions = {}

    log_message("Tracking started.")

    while not stop_tracking:    
        ret, frame = cap.read()
        if not ret:
            print("Video is finished or could not opened.")
            break

        display_width, display_height = 800, 450 #resize frame to fit GUI canvas
        frame = cv2.resize(frame, (display_width, display_height))

        detections = np.empty((0, 5)) #YOLO detection
        results = model(frame, stream=True)

        for info in results:
            boxes = info.boxes
            for box in boxes:
             x1, y1, x2, y2 = map(int, box.xyxy[0])
             conf = float(box.conf.item())
             classindex = int(box.cls.item())

            if 0 <= classindex < len(classnames):
                object_detected = classnames[classindex]
            else:
                continue

            if object_detected in ['car', 'truck', 'bus'] and conf > 0.6:
                new_detections = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, new_detections))

            cv2.imshow("Vehicle Detection",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
              break   

            cap.release()
            cv2.destroyAllWindows()

        track_result = tracker.update(detections)
        frame_time = 1 / FRAME_RATE

        for result in track_result:
            x1, y1, x2, y2, obj_id = map(int, result)
            center_x = (x1 + x2) // 2
            center_y = (y1+y2) // 2

            if obj_id in prev_positions:
                prev_x, prev_y, prev_time = prev_positions[obj_id]
                distance_pixels = np.sqrt((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2)
                speed_kmph = (distance_pixels * PIXEL_TO_METER) / frame_time * 3.6
            else:
                speed_kmph = 0

                prev_positions[obj_id] = (center_x, center_y, time.time())

                log_message(f"ID: {obj_id} | Class: {object_detected} | Speed: {speed_kmph:.1f} km/h")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) 
                cv2.putText(frame, f'ID: {obj_id} Speed: {speed_kmph:.1f} km/h', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2) 

                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


                try:
                    canvas.after(0, update_canvas, canvas, ImageTk.PhotoImage(Image.fromarray(img)))
                except:
                    continue

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                cap.release()
                cv2.destroyAllWindows()
                log_message("Tracking stopped.")

def update_canvas(canvas, img):
     """Update the canvas with the image."""
     canvas.create_image(0, 0, anchor=tk.NW, image=img)
     canvas.image = img


def stop_tracking_process():
     global stop_tracking
     stop_tracking = True
     log_message("Tracking stopped by user.")

def stop_monitoring():
     global stop_tracking
     stop_tracking = True
     root.destroy() 
     sys.exit(1)

def load_video():
    global video_path
    video_path = filedialog.askopenfilename(filetypes=[("video Files", "*.mp4;*.avi")])
    if video_path:
     messagebox.showinfo("Video Selected", f"Selected Video: {video_path}")
     print(f"Video selected: {video_path}")


root = tk.Tk()
root.title("Vehicle Speed Monitoring System")
root.geometry("900x700")
root.configure(bg="#f0f8ff")

style = ttk.Style()
style.configure(
    "TButton",
    font=("Helvetica" , 12, "bold"),
    foreground="#000000",
    background="#ff0000",
    padding=5
)

style.map(
    "TButton",
    background=[('active', '#cc0000')]
)

title_label = tk.Label(root, text="Vehicle Speed Monitoring System", font=("Helvetica", 20, "bold"), bg="#f0f8ff", fg="#333")
title_label.pack(pady=10)

canvas = tk.Canvas(root, width=800, height=450, bg="#dcdcdc")
canvas.pack(pady=10)

info_label = tk.Label(root, text="", font=("Helvetica", 12), bg="#f0f8ff", justify=tk.LEFT, anchor="w")
info_label.pack(pady=10, fill=tk.BOTH, padx=10)

button_frame = tk.Frame(root, bg="#f0f8ff")
button_frame.pack(pady=10)

load_video_button = ttk.Button(button_frame, text="load video", style="TButton", command=load_video)
load_video_button.grid(row=0, column=0, padx=5)

start_button = ttk.Button(button_frame, text="Start Tracking", style="TButton",
                          command=lambda: Thread(target=start_tracking, args=(canvas, info_label)).start())
start_button.grid(row=0, column=1, padx=5)

log_button = ttk.Button(button_frame, text="View logs", style="TButton", command=open_log_window)
log_button.grid(row=0, column=3, padx=5)

exit_button = ttk.Button(button_frame, text="Exit", style="TButton", command=stop_monitoring)
exit_button.grid(row=0, column=4, padx=5)

root.mainloop()

                
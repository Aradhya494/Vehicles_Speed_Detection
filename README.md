Vehicle Speed Detection System

A Python-based computer vision project that detects vehicles, tracks them across video frames, and estimates their real-time speed using YOLOv8 and SORT tracking.

Features

- Vehicle detection using YOLOv8.
- Vehicle tracking using a custom SORT tracking algorithm.
- Unique tracking IDs for detected vehicles.
- Real-time vehicle speed estimation.
- Pixel-to-meter calibration for speed calculation.
- Video processing using OpenCV.
- Simple GUI for selecting input video files using Tkinter.

Technologies Used

- Python
- OpenCV
- Ultralytics YOLOv8
- NumPy
- Pillow
- Tkinter
- SORT Tracking Algorithm

Project Structure

Vehicle-Speed-Detection/
│
├── vehicle_speed_detection.py
├── custom_sort.py
├── classes.txt
├── yolov8n.pt
└── README.md

File Description

- "vehicle_speed_detection.py" – Main Python script containing vehicle detection, tracking, and speed calculation.
- "custom_sort.py" – Custom implementation of the SORT tracking algorithm.
- "classes.txt" – Contains the names of detectable object classes.
- "yolov8n.pt" – Pre-trained YOLOv8 Nano model weights.
- "README.md" – Project documentation.

Installation

Make sure Python is installed on your system.

Install the required Python libraries using:

pip install opencv-python ultralytics numpy pillow

Tkinter is generally included with Python. If it is not available on your system, install it separately according to your operating system.

How to Run

1. Download or clone this repository.
2. Open the project folder in VS Code.
3. Open the terminal in the project directory.
4. Install the required dependencies.
5. Run the main Python file:

python vehicle_speed_detection.py

6. Select the traffic video when prompted.
7. The system will detect and track vehicles and calculate their estimated speed.

Working

The system works in the following steps:

Input Traffic Video
        ↓
YOLOv8 Vehicle Detection
        ↓
SORT Vehicle Tracking
        ↓
Vehicle ID Assignment
        ↓
Distance / Pixel Calibration
        ↓
Speed Calculation
        ↓
Vehicle Speed Output

Output

The system provides:

- Vehicle detection bounding boxes.
- Unique IDs for tracked vehicles.
- Estimated vehicle speed.
- Real-time vehicle tracking information.

Applications

This project can be useful for:

- Traffic monitoring.
- Vehicle speed analysis.
- Intelligent transportation systems.
- Road safety monitoring.
- Traffic management research.

Future Improvements

- Improve speed estimation accuracy.
- Add support for multiple camera views.
- Improve tracking in heavy traffic.
- Add a web-based dashboard for visualization.
- Store vehicle speed data for further analysis.

Author

Aradhya Rajhans

B.Tech Computer Science and Engineering Student

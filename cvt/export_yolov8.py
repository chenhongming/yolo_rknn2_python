from ultralytics import YOLO

# Load a model
model = YOLO('yolov8s.pt')  # load an official model

# Export the model
model.export(format='rknn',
             imgsz=[640, 640],
             simplify=True,
             opset=12,
             )

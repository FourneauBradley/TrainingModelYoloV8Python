import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_packages = ['tk', 'ultralytics', 'torch']

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        install(package)

import tkinter as tk
from tkinter.filedialog import *
from ultralytics import YOLO
import torch
import os
import threading
global button
def get_available_models():
    model_paths = []
    models_dir = './models'

    base_models = [
        ("No model",""),
        ("YOLOv8n (yolov8n.pt)", "yolov8n.pt"),
        ("YOLOv8s (yolov8s.pt)", "yolov8s.pt"),
        ("YOLOv8m (yolov8m.pt)", "yolov8m.pt"),
        ("YOLOv8l (yolov8l.pt)", "yolov8l.pt"),
        ("YOLOv8x (yolov8x.pt)", "yolov8x.pt")
    ]
    
    model_paths.extend(base_models)
    
    for root, dirs, files in os.walk(models_dir):
        if 'weights' in root:
            for file in files:
                if file.endswith('.pt'):
                    full_path = os.path.join(root, file)
                    display_name = f"{file} ({full_path})"
                    model_paths.append((display_name, full_path))
    
    return model_paths

def update_training_status(status):
    training_status.set(status)

def start_training_thread():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    selected_model = model_selection.get()
    model_path = model_selection_dict.get(selected_model, '')

    if model_path == '':
        print("Training without a pretrained model, initializing from scratch.")
        model = YOLO()
    else:
        print(f"Training with pretrained model: {model_path}")
        model = YOLO(model_path) 

    model.to(device)
    update_training_status("Training  in progress!")
    button.config(state="disabled")
    model.train(
        data=data_file_path.get(),
        epochs=epochs.get(),
        batch=batch_size.get(),
        imgsz=img_size.get(),
        lr0=0.01,
        device=device,
        project='models',
        name=modelName.get(),
        exist_ok=True,
        verbose=True
    )
    update_training_status("Training completed!")
    button.config(state="normal")

def start_training():
    thread = threading.Thread(target=start_training_thread)
    thread.start()

def select_file():
    file_path = askopenfilename(filetypes=[("Fichiers YAML", "*.yaml"), ("Tous les fichiers", "*.*")])
    data_file_path.set(file_path) 

root= tk.Tk()
root.title("Training model")
root.geometry("600x600")
tk.Label(root, text="Training model", font=("Arial", 30)).place(relx=0.05, rely=0.02, relwidth=0.9, height=50)
modelName=tk.StringVar()
epochs=tk.IntVar()
batch_size=tk.IntVar()
img_size=tk.StringVar()
data_file_path = tk.StringVar()
model_selection=tk.StringVar(value="yolov8n.pt")
model_options = get_available_models()
model_selection_dict = {name: path for name, path in model_options}

if model_options:
    model_selection.set(model_options[0][0])
fields = [
    ("Model name :",modelName),
    ("Epochs :",epochs),
    ("Batch size :",batch_size),
    ("Image size :",img_size)
]
epochs.set(10)
batch_size.set(8)
img_size.set("800")
label_x = 0.05
entry_x = 0.4
row_height = 40
for i, (label, var) in enumerate(fields):
    y_position = 80 + i * row_height
    tk.Label(root, text=label, anchor='w').place(relx=label_x, y=y_position, relwidth=0.3, height=30)
    tk.Entry(root, textvariable=var).place(relx=entry_x, y=y_position, relwidth=0.5, height=30)

y_position += row_height + 20
tk.Label(root, text="Data file (data.yaml) :", anchor='w').place(relx=label_x, y=y_position, relwidth=0.3, height=30)
tk.Entry(root, textvariable=data_file_path).place(relx=entry_x, y=y_position, relwidth=0.5, height=30)
tk.Button(root, text="Browse", command=select_file).place(relx=0.9, y=y_position, anchor='ne')

y_position += row_height + 20
tk.Label(root, text="Select base model :", anchor='w').place(relx=label_x, y=y_position, relwidth=0.3, height=30)
model_menu = tk.OptionMenu(root, model_selection, *([name for name, path in model_options]))
model_menu.place(relx=entry_x, y=y_position, relwidth=0.5, height=30)


y_position += row_height + 20
training_status = tk.StringVar()
training_status.set("Ready to start training.")
tk.Label(root, text="Training Status:", anchor='w').place(relx=label_x, y=y_position, relwidth=0.3, height=30)
tk.Label(root, textvariable=training_status).place(relx=entry_x, y=y_position, relwidth=0.5, height=30)

button_y_position = y_position + row_height + 20
button=tk.Button(root, text="Start Training", command=start_training)
button.place(relx=0.5, y=button_y_position, anchor='center')
root.mainloop()
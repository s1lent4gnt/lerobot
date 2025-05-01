import os
import cv2
import csv

# === Configuration ===
frames_dir = "output_frames/video_frames"  # folder where PNGs are saved
labels_csv = "output_frames/intervention_labels.csv"
output_video = "output_frames/annotated_video.mp4"
framerate = 10  # original capture fps
border_thickness = 6
font = cv2.FONT_HERSHEY_SIMPLEX

# === Load labels ===
labels = []
with open(labels_csv, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        labels.append((row['frame'], int(row['label'])))

# === Get frame size ===
first_frame = cv2.imread(os.path.join(frames_dir, labels[0][0]))
height, width, _ = first_frame.shape

# === Initialize video writer ===
out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), framerate, (width, height))

# === Process each frame ===
for i, (filename, label) in enumerate(labels):
    path = os.path.join(frames_dir, filename)
    frame = cv2.imread(path)

    # 1. Draw border
    color = (0, 255, 0) if label == 0 else (0, 0, 255)  # green or red
    cv2.rectangle(frame, (0, 0), (width - 1, height - 1), color, thickness=border_thickness)

    # 2. Draw timestamp
    time_sec = i / framerate
    time_str = f"{time_sec:.2f} sec"
    cv2.putText(frame, time_str, (30, 60), font, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

    # 3. Write to video
    out.write(frame)

out.release()
print(f"[✓] Video saved to: {output_video}")
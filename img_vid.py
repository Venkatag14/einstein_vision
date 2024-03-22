import os
import cv2

def extract_frames_from_video(video_path, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    # Read frames until the video is finished
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame as an image
        if frame_count%10 == 0:
            frame_output_path = os.path.join(output_folder, f"{int(frame_count/10)}.jpg")
            cv2.imwrite(frame_output_path, frame)

        # Increment frame count
        frame_count += 1

    # Release video capture object
    cap.release()

# Example usage:
video_path = r"Sequences\scene4\Raw\2023-02-14_11-51-54-front.mp4"
output_folder = r"scene4\front_frames"

extract_frames_from_video(video_path, output_folder)

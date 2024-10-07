import cv2


class SaveVideo:
    def __init__(self, output_video_path, frame_width, frame_height, fps):
        self.output_video_path = output_video_path
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can use other codecs like 'MJPG', 'X264', etc.
        self.out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    def add_frame(self, frame):
        self.out.write(frame)

    def close(self):
        self.out.release()
        print(f"Video saved as {self.output_video_path}")


def frame_generator(video_path, batch_size=30, frame_intervals=2):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not video.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        frames = []
        frame_count = 0
        for _ in range(batch_size):
            ret, frame = video.read()
            if not ret:
                break
            if frame_count % frame_intervals == 0:
                frames.append(frame)
            frame_count += 1

        if frames:
            yield frames
            frames.clear()  # Clear the list after yielding
        else:
            break

    # Release the video capture object
    video.release()
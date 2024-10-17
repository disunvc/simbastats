import cv2
import numpy as np
from utils.video import SaveVideo, frame_generator
from utils.my_tracker import Tracker
from utils.assign_teams import TeamAssigner

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('video/result_output.avi', fourcc, 25, (1280, 720))


def apply_birds_eye_view(frame, src_points, dest_points):
    """
    Apply a perspective transformation to get a bird's-eye view.

    :param frame: The input video frame
    :param src_points: Source points on the frame to be transformed
    :param dest_points: Destination points for the transformation
    :return: Transformed bird's-eye view frame
    """
    # Calculate the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dest_points)

    # Warp the frame to the new perspective
    bird_eye_frame = cv2.warpPerspective(frame, matrix, (frame.shape[1], frame.shape[0]))
    return bird_eye_frame


def main():
    video_path = 'video/case_study_video1.mov'
    tracker = Tracker('models/football.pt')

    # Define the points for the perspective transformation
    # Adjust these points according to your video and desired bird's-eye view
    src_points = np.float32([[100, 100], [1180, 100], [1180, 620], [100, 620]])  # Example source points
    dest_points = np.float32([[0, 0], [1280, 0], [1280, 720], [0, 720]])  # Full frame as destination points

    # Using the generator to get batches of frames
    frame_gen = frame_generator(video_path)

    frame_num = 0
    for bs, frame_batch in enumerate(frame_gen):
        for img in frame_batch:
            try:
                # Skip empty frames
                if img is None or img.size == 0:
                    print(f"Empty frame encountered at frame number: {frame_num}")
                    frame_num += 1
                    continue

                # Track objects in the frame
                frame, tracks = tracker.get_object_tracks(
                    img, frame_num, read_from_stub=False, stub_path='models/track_stubs.pkl'
                )
                frame_num += 1

                # Apply bird's-eye view transformation
                bird_eye_frame = apply_birds_eye_view(frame, src_points, dest_points)

                # Display the transformed frame
                cv2.imshow('Bird-Eye View Frame', bird_eye_frame)

                # Save the frame to the video file
                out.write(bird_eye_frame)

                # Break on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except Exception as e:
                print(f"Error processing frame {frame_num}: {e}")
                continue

    # Release resources
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

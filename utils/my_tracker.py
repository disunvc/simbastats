from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
from collections import defaultdict
from utils.bbox import get_center_of_bbox, get_bbox_width, get_foot_position,measure_distance
from utils.assign_teams import TeamAssigner
from utils.df_write import create_and_write_csv
from filterpy.kalman import KalmanFilter


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoundingBoxAnnotator()
        self.team_assigner = TeamAssigner()
        self.init_team_assigner = 0
        self.add_row_to_csv = create_and_write_csv('result_data.csv')
        self.color_name = None
        self.kalman_filters = {}  # Dictionary to hold Kalman filters for each track
        self.confidence_scores = {}  # Dictionary to hold confidence scores
        self.yolo_to_kalman_map = {}  # Mapping YOLO IDs to Kalman track IDs
        self.threshold_distance = 30

    def create_kalman_filter(self, track_id):
        """Initialize a Kalman filter for tracking."""
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.x = np.array([[0], [0], [0], [0]])  # State vector (x, y, vx, vy)
        kf.P *= 1000  # Initial uncertainty
        kf.F = np.array([[1, 0, 1, 0],  # State transition matrix
                         [0, 1, 0, 1],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0],  # Measurement function
                         [0, 1, 0, 0]])
        kf.R = np.array([[10, 0],       # Measurement noise
                         [0, 10]])
        kf.Q = np.eye(4)  # Process noise
        self.kalman_filters[track_id] = kf
        self.confidence_scores[track_id] = 5  # Initialize with high confidence score

    def update_kalman_filter(self, track_id, measurement):
        """Update Kalman filter with new measurement."""
        kf = self.kalman_filters[track_id]
        kf.predict()
        kf.update(measurement)
        return kf.x

    def calculate_confidence_score(self, track_id, measurement):
        """Calculate a confidence score based on tracking stability."""
        kf = self.kalman_filters[track_id]
        prediction_error = np.linalg.norm(measurement - kf.x[:2])
        confidence = max(1, 5 - int(prediction_error / 30))  # Adjust scaling factor to reduce strictness
        self.confidence_scores[track_id] = confidence
        return confidence

    def detect_frames(self, frames):
        try:
            results = self.model.predict(frames, conf=0.08, verbose=False)[0]
            if not results:
                raise ValueError("No frames detected.")
            return results
        except Exception as e:
            print(f"Error detecting frames: {e}")
            return []

    def get_object_tracks(self, frames, frame_count, read_from_stub=False, stub_path=None):
        try:
            # Ensure frames are valid
            if frames is None:
                raise ValueError(f"Frame {frame_count} is None. Skipping this frame.")

            # Read from stub if required
            if read_from_stub and stub_path is not None and os.path.exists(stub_path):
                with open(stub_path, 'rb') as f:
                    tracks = pickle.load(f)
                return tracks

            # YOLO detection
            yolo_results = self.detect_frames(frames)
            if not yolo_results:
                raise ValueError("YOLO detection returned no results.")

            detection_supervision = sv.Detections.from_ultralytics(yolo_results)
            tracks = {
                "players": [],
                "referees": [],
                "ball": [],
                "goalkeepers": []
            }

            frame_num = 0
            for _, detection in enumerate(yolo_results):
                cls_names = detection.names
                cls_names_inv = {v: k for k, v in cls_names.items()}

                required_classes = ['player', 'referee', 'ball', 'goalkeeper']
                for cls in required_classes:
                    if cls not in cls_names_inv:
                        print(f"Warning: Class '{cls}' not found in model results.")

                for object_ind, class_id in enumerate(detection_supervision.class_id):
                    if cls_names[class_id] == "goalkeeper":
                        detection_supervision.class_id[object_ind] = cls_names_inv["player"]

                detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

                tracks["players"].append({})
                tracks["referees"].append({})
                tracks["ball"].append({})
                tracks["goalkeepers"].append({})

                for dc, frame_detection in enumerate(detection_with_tracks):
                    bbox = frame_detection[0].tolist()
                    cls_id = frame_detection[3]
                    track_id = frame_detection[4]
                    yolo_id = dc  # Assuming YOLO assigns incremental IDs starting from 0

                    # Update mapping of YOLO ID to Kalman track ID
                    self.yolo_to_kalman_map[yolo_id] = track_id

                    if track_id not in self.kalman_filters:
                        self.create_kalman_filter(track_id)

                    measurement = np.array([bbox[0], bbox[1]])
                    self.update_kalman_filter(track_id, measurement)

                    if cls_id == cls_names_inv['player']:
                        tracks["players"][frame_num][track_id] = {"bbox": bbox, "yolo_id": yolo_id}
                    elif cls_id == cls_names_inv['referee']:
                        tracks["referees"][frame_num][track_id] = {"bbox": bbox, "yolo_id": yolo_id}
                    elif cls_id == cls_names_inv['goalkeeper']:
                        tracks["goalkeepers"][frame_num][track_id] = {"bbox": bbox, "yolo_id": yolo_id}

                for frame_detection in detection_supervision:
                    bbox = frame_detection[0].tolist()
                    cls_id = frame_detection[3]

                    if cls_id == cls_names_inv['ball']:
                        tracks["ball"][frame_num][1] = {"bbox": bbox}

                img = frames.copy()
                if self.init_team_assigner == 0:
                    self.team_assigner.assign_team_color(img, tracks['players'][0])
                self.init_team_assigner = 1

                for fn, player_track in enumerate(tracks['players']):
                    for player_id, track in player_track.items():
                        team = self.team_assigner.get_player_team(img, track['bbox'], player_id)
                        tracks['players'][fn][player_id]['team'] = team
                        tracks['players'][fn][player_id]['team_color'] = self.team_assigner.team_colors[team]

                track_frame = img.copy()
                player_dict = tracks["players"][frame_num]
                ball_dict = tracks["ball"][frame_num]
                referee_dict = tracks["referees"][frame_num]
                goalkeeper_dict = tracks["goalkeepers"][frame_num]

                # Pass Detection Logic
                ball_position = None
                if ball_dict and 1 in ball_dict:  # Make sure there is ball data for this frame
                    ball_position = get_center_of_bbox(ball_dict[1]["bbox"])

                if ball_position:
                    for track_id, player in player_dict.items():
                        player_position = get_foot_position(player["bbox"])

                        # Measure distance between player and ball
                        distance_to_ball = measure_distance(player_position, ball_position)

                        # Detect if a player has possession (pass detection)
                        if distance_to_ball < self.threshold_distance:
                            if self.previous_ball_holder is not None and self.previous_ball_holder != track_id:
                                # A pass is detected
                                print(
                                    f"Pass detected from Player {self.previous_ball_holder} to Player {track_id} at Frame {frame_num}")

                                # Log the pass event to the CSV
                                self.add_row_to_csv(frame_num, self.previous_ball_holder, track_id, player_position[0],
                                                    player_position[1], 'Pass', object_type="Pass")

                            # Update ball possession to the current player
                            self.previous_ball_holder = track_id
                            print(f"Ball now with Player {track_id} at Frame {frame_num}")

                        # Log the player's position normally if no pass is detected
                        else:
                            self.add_row_to_csv(frame_num, track_id, track_id, player_position[0], player_position[1],
                                                player.get('team_color', 'unknown'), object_type="Player")

                for track_id, player in player_dict.items():
                    color = player.get("team_color", (0, 0, 255))
                    track_frame = self.draw_track_ellipse(track_frame, player["bbox"], color, track_id)

                    cx, cy = get_foot_position(player["bbox"])
                    color_array = list(color)
                    self.color_name = 'red' if color_array == [255, 0, 0] else 'blue'
                    yolo_id = player.get("yolo_id", 'Unknown')

                    # Write both YOLO and Kalman IDs to CSV
                    self.add_row_to_csv(frame_count, yolo_id, track_id, cx, cy, self.color_name, object_type="Player")

                    if player.get('has_ball', False):
                        track_frame = self.draw_traingle(track_frame, player["bbox"], (0, 0, 255))

                for _, referee in referee_dict.items():
                    cx, cy = get_foot_position(referee["bbox"])
                    yolo_id = referee.get("yolo_id", 'Unknown')
                    # Add or update the row with 'Referee' object type
                    track_frame = self.draw_track_ellipse(track_frame, referee["bbox"], (255, 167, 255))
                    self.add_row_to_csv(frame_count, yolo_id, track_id, cx, cy, "yellow", object_type="Referee")

                for track_id, goalkeeper in goalkeeper_dict.items():
                    cx, cy = get_foot_position(goalkeeper["bbox"])
                    yolo_id = goalkeeper.get("yolo_id", 'Unknown')
                    track_frame = self.draw_track_ellipse(track_frame, referee["bbox"], (167, 255, 167))
                    # Add or update the row with 'Goalkeeper' object type
                    self.add_row_to_csv(frame_count, yolo_id, track_id, cx, cy, "blue", object_type="Goalkeeper")

                for track_id, ball in ball_dict.items():
                    track_frame = self.draw_traingle(track_frame, ball["bbox"], (0, 255, 0))
                    cv2.putText(track_frame, f'Ball {track_id}',
                                (int(ball["bbox"][0]), int(ball["bbox"][1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    self.add_row_to_csv(frame_count, yolo_id, track_id, cx, cy, "yellow", object_type="Ball")

                frame_num += 1

                if stub_path is not None:
                    with open(stub_path, 'wb') as f:
                        pickle.dump(tracks, f)

                return track_frame, tracks

        except Exception as e:
            print(f"Error in get_object_tracks: {e}")
            return None, {}

    def draw_track_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.1 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=5,
            lineType=cv2.LINE_4
        )

        if track_id is not None:
            cv2.putText(frame, f'ID {track_id}', (x_center - 20, y2 - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return frame

    def draw_traingle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

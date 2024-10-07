from sklearn.cluster import KMeans
import cv2
import numpy as np


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

    def classify_teams(self, player, color):
        player = cv2.cvtColor(player, cv2.COLOR_BGR2HSV)
        masks = []
        Hsv_boundaries = [
            ([0, 0, 185], [180, 80, 255], 'white'),  # white
            ([0, 0, 0], [180, 255, 65], 'black'),  # black
            ([0, 120, 20], [15, 255, 255], 'red'),  # red
            ([163, 100, 20], [180, 255, 255], 'red'),  # dark red/pink
            ([90, 100, 20], [130, 255, 255], 'blue'),  # blue
            ([50, 100, 20], [80, 255, 255], 'green'),  # green
            ([17, 150, 20], [35, 255, 255], 'yellow'),  # yellow
            ([131, 60, 20], [162, 255, 255], 'purple'),  # purple
            ([80, 36, 20], [105, 255, 255], 'skyblue'),  # skyblue
            ([10, 120, 20], [20, 255, 255], 'orange'),  # orange - added for the current game
            ([40, 120, 20], [70, 255, 255], 'lime'),  # lime
            ([5, 50, 50], [15, 255, 255], 'brown'),  # brown
        ]

        filtered_boundaries = list(filter(lambda boundary: boundary[2] in color, Hsv_boundaries))
        for boundary in filtered_boundaries:
            mask = cv2.inRange(player, np.array(boundary[0]), np.array(boundary[1]))
            count = np.count_nonzero(cv2.bitwise_and(player, player, mask=mask))
            masks.append(count)

        color_idx = masks.index(max(masks))
        return filtered_boundaries[color_idx][2]

    def get_clustering_model(self, image):
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(image_2d)
        return kmeans

    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half_image = image[0:int(image.shape[0] / 2), :]

        kmeans = self.get_clustering_model(top_half_image)
        labels = kmeans.labels_
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            player_color = self.classify_teams(player, ['red', 'blue', 'white', 'orange'])
            if player_color == 'red':
                player_color = np.array([255, 0, 0])
            elif player_color == 'blue':
                player_color = np.array([0, 0, 255])
            elif player_color == 'white':
                player_color = np.array([255, 255, 255])
            elif player_color == 'orange':
                player_color = np.array([255, 165, 0])
            else:
                player_color = np.array([128, 128, 128])  # Default gray

            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=5, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0] + 1

        self.player_team_dict[player_id] = team_id
        return team_id
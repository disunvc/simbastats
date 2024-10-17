import pandas as pd

def create_and_write_csv(file_name):
    # Define the column names, including track_id to separate YOLO and Kalman filter IDs
    columns = ['frame_count', 'yolo_id', 'track_id', 'x', 'y', 'color', 'object_type']

    # Create an empty DataFrame with these column names
    df = pd.DataFrame(columns=columns)

    # Save the empty DataFrame to a CSV file
    df.to_csv(file_name, index=False)

    # Function to add a row to the CSV
    def add_row(frame_count, yolo_id, track_id, x, y, color, object_type):
        # Create a new row as a dictionary
        row = {
            'frame_count': frame_count,
            'yolo_id': yolo_id if yolo_id is not None else 'N/A',  # Handle cases when YOLO ID is missing
            'track_id': track_id,  # Kalman filter track ID
            'x': x,
            'y': y,
            'color': color,
            'object_type': object_type  # e.g., 'Player', 'Pass', 'Ball', etc.
        }


        # Append the row to the DataFrame
        df_new = pd.DataFrame([row])

        # Append the new DataFrame to the CSV file without rewriting the header
        df_new.to_csv(file_name, mode='a', header=False, index=False)

    # Return the function to add rows
    return add_row

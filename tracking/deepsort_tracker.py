from deep_sort_realtime.deepsort_tracker import DeepSort

class DeepSORTTracker:
    def __init__(self):
        self.tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)

    def track_objects(self, detections, frame):
        # Pass only filtered detections to DeepSORT
        tracks = self.tracker.update_tracks(detections, frame=frame)
        return tracks

import cv2

# Define colors for tracked objects
CLASS_COLORS = {
    "car": (0, 255, 0),  # Green
    "truck": (255, 0, 0),  # Red
    "motorcycle": (0, 0, 255),  # Blue
    "bus": (0, 255, 255),  # Yellow
}

def draw_boxes(image, tracks, class_names):
    for track in tracks:
        if track.is_confirmed() and track.time_since_update == 0:
            x, y, w, h = track.to_tlwh()
            track_id = track.track_id
            class_name = class_names.get(track.det_class, "object")

            # Only display the required classes
            if class_name in CLASS_COLORS:
                color = CLASS_COLORS[class_name]
                cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, 1)

                # Create label with class name and confidence
                label = f"{class_name}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

                # Draw filled rectangle for label background
                cv2.rectangle(image, (int(x), int(y - th - 3)), (int(x) + tw + 6, int(y)), color, -1)

                # Draw text label on top
                cv2.putText(image, label, (int(x) + 2, int(y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)  # Black text for contrast

    return image

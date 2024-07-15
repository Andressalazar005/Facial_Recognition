import cv2
import face_recognition
import numpy as np
import mss

def draw_label(image, text, pos, bg_color, text_color):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    margin = 5

    size = cv2.getTextSize(text, font_face, scale, thickness)
    end_x = pos[0] + size[0][0] + margin
    end_y = pos[1] - size[0][1] - margin

    cv2.rectangle(image, pos, (end_x, end_y), bg_color, cv2.FILLED)
    cv2.putText(image, text, (pos[0], pos[1] - 5), font_face, scale, text_color, thickness, cv2.LINE_AA)

def main():
    # Initialize the screen capture object
    sct = mss.mss()
    
    # Define the screen capture area (full screen)
    monitor = sct.monitors[1]

    while True:
        # Capture the screen
        screenshot = sct.grab(monitor)
        
        # Convert the screenshot to a numpy array
        frame = np.array(screenshot)

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Convert the frame from BGRA to RGB
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGRA2RGB)

        # Find all face locations in the frame using the cnn model
        face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn")

        # Scale back up face locations since we detected them on a smaller frame
        face_locations = [(top*2, right*2, bottom*2, left*2) for top, right, bottom, left in face_locations]

        for (top, right, bottom, left) in face_locations:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            # Add a label with the text 'Face'
            draw_label(frame, 'Face', (left, top), (0, 0, 0), (255, 255, 255))

        # Display the resulting frame
        cv2.imshow('Screen', frame)

        # Exit the video display window when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close any OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

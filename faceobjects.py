import cv2
import numpy as np
import mss
import os
import time
from concurrent.futures import ThreadPoolExecutor

# Load YOLO
def load_yolo():
    yolo_path = os.path.join('yolo', 'yolov4-tiny.cfg')
    weights_path = os.path.join('yolo', 'yolov4-tiny.weights')
    names_path = os.path.join('yolo', 'coco.names')
    
    net = cv2.dnn.readNetFromDarknet(yolo_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

# Load Haar cascades
def load_cascades(cascade_dir):
    cascades = {}
    for filename in os.listdir(cascade_dir):
        if filename.endswith('.xml'):
            cascade_name = filename.split('.')[0]
            cascades[cascade_name] = cv2.CascadeClassifier(os.path.join(cascade_dir, filename))
    return cascades

def draw_label(image, text, pos, bg_color, text_color):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 1
    margin = 5

    size = cv2.getTextSize(text, font_face, scale, thickness)
    end_x = pos[0] + size[0][0] + margin
    end_y = pos[1] - size[0][1] - margin

    cv2.rectangle(image, pos, (end_x, end_y), bg_color, cv2.FILLED)
    cv2.putText(image, text, (pos[0], pos[1] - 5), font_face, scale, text_color, thickness, cv2.LINE_AA)

def detect_objects(frame, net, output_layers):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return boxes, confidences, class_ids, indexes

def detect_faces(frame, person_boxes, cascades, min_confidence=1.0):
    faces = []
    for (x, y, w, h) in person_boxes:
        person_roi = frame[y:y+h, x:x+w]
        for name, cascade in cascades.items():
            detected = cascade.detectMultiScale(person_roi, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (fx, fy, fw, fh) in detected:
                face_confidence = 1.0  # Assuming equal weightage for all cascades
                faces.append((x + fx, y + fy, fw, fh, face_confidence, name))
    
    if len(faces) == 0:
        return []

    # Apply non-maximum suppression to faces
    face_boxes = np.array([box[:4] for box in faces])
    face_confidences = np.array([box[4] for box in faces])
    face_indexes = cv2.dnn.NMSBoxes(face_boxes.tolist(), face_confidences.tolist(), score_threshold=0.5, nms_threshold=0.4)

    if isinstance(face_indexes, tuple):
        face_indexes = face_indexes[0]

    if len(face_indexes) == 0:
        return []

    filtered_faces = [faces[i] for i in face_indexes.flatten()]
    return filtered_faces

def process_frame(frame, net, output_layers, classes, cascades):
    height, width, channels = frame.shape
    boxes, confidences, class_ids, indexes = detect_objects(frame, net, output_layers)

    person_boxes = []
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            if label == "person":
                person_boxes.append((x, y, w, h))
                color = (0, 255, 255)  # Yellow for person boxes
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                draw_label(frame, f'{label}: {confidence:.2f}', (x, y), (0, 0, 0), (255, 255, 255))
            else:
                color = (0, 0, 255)  # Red for other objects
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                draw_label(frame, f'{label}: {confidence:.2f}', (x, y), (0, 0, 0), (255, 255, 255))

    filtered_faces = detect_faces(frame, person_boxes, cascades, min_confidence=1.0)

    for (x, y, w, h, confidence, cascade_name) in filtered_faces:
        color = (0, 255, 0)  # Green for face boxes
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        draw_label(frame, 'Face', (x, y), (0, 0, 0), (255, 255, 255))

    return frame

def main():
    net, classes, output_layers = load_yolo()
    cascades = load_cascades('cascades')

    sct = mss.mss()
    monitor = sct.monitors[1]

    with ThreadPoolExecutor() as executor:
        while True:
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            start_time = time.time()

            future = executor.submit(process_frame, frame, net, output_layers, classes, cascades)
            processed_frame = future.result()

            fps = 1.0 / (time.time() - start_time)
            draw_label(processed_frame, f'FPS: {fps:.2f}', (10, 30), (0, 0, 0), (255, 255, 255))

            cv2.imshow('Screen', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

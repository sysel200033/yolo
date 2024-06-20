import cv2
from ultralytics import YOLO

yolo = YOLO('yolov8s.pt')
videoCap = cv2.VideoCapture("./images/people.mp4")

# Function to get class colors
def getColors(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] * 
    (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)


while True:
    ret, frame = videoCap.read()
    # cv2.resize(frame, (1020, 720))
    # ret tells if frame was read successfully or not
    # frame numpy array the holds video frame
    if not ret:
        continue
    results = yolo.track(frame, stream=True)

    for result in results:
        # get the classes names for each detected Object an Id
        classes_names = result.names

        # iterate over each box bounding boxes for the detected objects represented by a set of coordinates (x1, y1, x2, y2)
        for box in result.boxes:
            # check if confidence (Confidence Score tells the models confidence in the detection --> higher more confident) is greater than 40 percent
            if box.conf[0] > 0.4:
                # get coordinates
                [x1, y1, x2, y2] = box.xyxy[0]
                # convert to int
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # get the class of the detected object
                cls = int(box.cls[0])

                # get the class name
                class_name = classes_names[cls]

                # get the respective color (unique color for every object)
                color = getColors(cls) 

                # draw rectangle in video frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2) # frame, top-left, bottom-right, color, thickness of border

                # put the class name and confidence on the image
                # frame, text and confidence, postition of text, font style, font scale, color, thickness of text lines
                cv2.putText(frame, f'{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2) 
                
    # show the frame of the video
    cv2.imshow('frame', frame)

    # break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the video capture and destroy all windows
videoCap.release()
cv2.destroyAllWindows()
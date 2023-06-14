import os
import cv2


DIR = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.join(DIR, "MobileNetSSD_deploy.prototxt")
WEIGHTS = os.path.join(DIR, "MobileNetSSD_deploy.caffemodel")

classNames = { 0: 'background',
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }
     

def identify_objects(img_path_file, output_dir):
    img = cv2.imread(img_path_file)
    
    img_resized = cv2.resize(img , (300 , 300))
    blob = cv2.dnn.blobFromImage(
        img_resized,
        0.007843,
        (300, 300),
        (127.5, 127.5, 127.5), 
        False
    )

    # Get from trained base
    net = cv2.dnn.readNetFromCaffe(PROTO , WEIGHTS)
    net.setInput(blob)

    # Detect objects
    detections = net.forward()
    detection = detections.squeeze()

    image = generate_image(img, detection)
    output_filename = os.path.join(output_dir, "ssd-result.jpg")
    cv2.imwrite(output_filename, image)
    print(f"Resultado salvo em: {output_filename}")


def generate_image(img, detection):
    height , width , _ = img.shape
    detection_height = detection.shape[0]
    
    for i in range(detection_height):
        conf = detection[i , 2]
        if conf > 0.5:
            class_name = classNames[detection[i , 1]]
            x1 , y1 , x2 , y2 = detection[i , 3:]
            x1 *= width
            y1 *= height
            x2 *= width
            y2 *= height
            top_left = (int(x1) , int(y1))
            bottom_right = (int(x2) , int(y2))
            img = cv2.rectangle(img, top_left , bottom_right , (0 , 255 , 0) , 3)
            img = cv2.putText(img, class_name , top_left, cv2.FONT_HERSHEY_SIMPLEX , 
                            1 , (255 , 0 , 0) , 2 , cv2.LINE_AA)    

    return img
    
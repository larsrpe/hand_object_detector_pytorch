from cv2 import imread
from handobdet import HandObjectDetector, draw_detections

def main():
    input = imread("/Users/larsrpe/hand_object_detector_pytorch/images/5137daf7b29c5f4bc37322122803e597.jpg")
    detector = HandObjectDetector(input_format="BGR")
    dets=detector.detect(input,viz=False)
if __name__ == "__main__":
    main()

from torchvision.io import read_image
from src.hand_object_detector import HandObjectDetector

def main():
    input = read_image("images/5137daf7b29c5f4bc37322122803e597.jpg")
    detector = HandObjectDetector()
    detector.detect(input,viz=True)

if __name__ == "__main__":
    main()

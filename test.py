from src.antigener_detection.antigener_detector import antigener_classification_update_1, Searcher
import cv2

searcher, id_map = Searcher()

img = cv2.imread('./tests/test_images/FIHmHpwWQAcKWK3.jpg')

results = antigener_classification_update_1(img, searcher, id_map)

print(results)
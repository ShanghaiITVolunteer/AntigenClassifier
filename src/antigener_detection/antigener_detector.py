import cv2

from detector import DetPredictor
from detutils.config import get_config


"""
修改成绝对路径。
"""
modelparams_path = "pretrained_model/hansansui_antigen_detector"
config_path = "pretrained_model/hansansui_antigen_detector/infer_cfg.yml"

config = get_config(config_path,
                    show=False)
antigener_detector = DetPredictor(modelparams_path,
                                  config)

def postprocess_after_detection(roi_img, count=0, binary_threshed=130):
    roi_img = cv2.GaussianBlur(roi_img, (3, 3), 0)
    roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    _, threshed_img = cv2.threshold(roi_img, binary_threshed, 255, 0)
    contours, _ = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    new_contours = []
    for contour in contours:
        #print(cv2.contourArea(contour))
        if cv2.contourArea(contour) > 500:
            rect = cv2.minAreaRect(contour)
            w, h = rect[1][0], rect[1][1]
            #print(h / w, w / h)
            if (3.5 >= w / h >= 1.8 ) or (3.5 >= h / w >= 1.8):
                new_contours.append(contour)
    #print(len(new_contours))
    #cv2.namedWindow('threshed{}'.format(count), 0)
    #cv2.imshow("threshed{}".format(count), threshed_img)

    return len(new_contours)

def antigener_classification(img, det_threshed=0.2, negative_threshed=0.65):

    results = antigener_detector.predict(img, det_threshed)
    count = 0
    results_dict = {'Positive':[], 'Negative':[]}
    for result in results:
        count += 1
        x1, y1, x2, y2 = result['bbox']
        roi_img = img[int(y1):int(y2), int(x1):int(x2)]
        if result['label_name'] == 'positive':
            """
            输出为positive+confidence.
            经后处理如果也认定阳性，输出为positive, confidence为1.
            """
            length = postprocess_after_detection(roi_img, count)
            if length == 2:
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                cv2.putText(img, 'Positive', (int(x1), int(y1)), 0, 0.7, (0, 0, 255), 2)
                cv2.putText(img, '1.0', (int(x1), int(y1-30)), 0, 0.7, (0, 0, 255), 2)
                results_dict['Positive'].append([x1, y1, x2, y2, 1.0])
            else:
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                cv2.putText(img, 'Positive', (int(x1), int(y1)), 0, 0.7, (0, 0, 255), 2)
                cv2.putText(img, '{}'.format(result['score']), (int(x1), int(y1-30)), 0, 0.7, (0, 0, 255), 2)
                results_dict['Positive'].append([x1, y1, x2, y2, result['score']])

        else: #阴性
            if result['score'] > negative_threshed:

                length = postprocess_after_detection(roi_img, count)
                if length == 1:
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                    cv2.putText(img, 'Negative', (int(x1), int(y1)), 0, 0.7, (0, 255, 0), 2)
                    cv2.putText(img, '1.0', (int(x1), int(y1-30)), 0, 0.7, (0, 255, 0), 2)
                    results_dict['Negative'].append([x1, y1, x2, y2, 1.0])
                else:
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                    cv2.putText(img, 'Negative', (int(x1), int(y1)), 0, 0.7, (0, 255, 0), 2)
                    cv2.putText(img, '{}'.format(result['score']), (int(x1), int(y1-30)), 0, 0.7, (0, 255, 0), 2)
                    results_dict['Negative'].append([x1, y1, x2, y2, result['score']])
            else:
                """
                小于这个阈值，输出为positive，confidence为1-confidence。
                经后处理后若认为为positive, confidence为1。
                """
                length = postprocess_after_detection(roi_img, count)
                if length == 2:
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                    cv2.putText(img, 'Positive', (int(x1), int(y1)), 0, 0.7, (0, 0, 255), 2)
                    cv2.putText(img, '1.0', (int(x1), int(y1-30)), 0, 0.7, (0, 0, 255), 2)
                    results_dict['Positive'].append([x1, y1, x2, y2, 1.0])
                else:
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                    cv2.putText(img, 'Positive', (int(x1), int(y1)), 0, 0.7, (0, 0, 255), 2)
                    cv2.putText(img, '{}'.format(1-result['score']), (int(x1), int(y1 - 30)), 0, 0.7, (0, 0, 255), 2)
                    results_dict['Positive'].append([x1, y1, x2, y2, 1 - result['score']])

    #print(result)
    cv2.imwrite('tests/test_images/detect_output.png', img)
    cv2.namedWindow('img', 0)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    return results_dict

if __name__ == "__main__":
    img = cv2.imread('tests/test_images/positive.jpeg')
    results = antigener_classification(img)
    print(results)

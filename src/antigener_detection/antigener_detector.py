import cv2
import os
import faiss
import pickle
import numpy as np

from recognition import RecPredictor
from detector import DetPredictor
from utils.config import get_config


"""
修改成绝对路径。
"""
modelparams_path = "./params/detection_model"
config_path = "./params/infer_cfg.yml"

config = get_config(config_path,
                    show=False)

rec_predictor = RecPredictor(config)
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
"""
###############################################
###############################################
################2022.5.11 update###############
###############################################
###############################################
"""

def Searcher():
    assert 'IndexProcess' in config.keys(), "Index config not found ... "

    index_dir = config["IndexProcess"]["index_dir"]
    assert os.path.exists(os.path.join(
        index_dir, "vector.index")), "vector.index not found ..."
    assert os.path.exists(os.path.join(
        index_dir, "id_map.pkl")), "id_map.pkl not found ... "

    if config['IndexProcess'].get("binary_index", False):
        searcher = faiss.read_index_binary(
            os.path.join(index_dir, "vector.index"))
    else:
        searcher = faiss.read_index(
            os.path.join(index_dir, "vector.index"))

    with open(os.path.join(index_dir, "id_map.pkl"), "rb") as fd:
        id_map = pickle.load(fd)
    return searcher, id_map

def nms_to_rec_results(results, thresh=0.05):
    filtered_results = []
    x1 = np.array([r["bbox"][0] for r in results]).astype("float32")
    y1 = np.array([r["bbox"][1] for r in results]).astype("float32")
    x2 = np.array([r["bbox"][2] for r in results]).astype("float32")
    y2 = np.array([r["bbox"][3] for r in results]).astype("float32")
    scores = np.array([r["rec_scores"] for r in results])

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    while order.size > 0:
        i = order[0]
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
        filtered_results.append(results[i])

    return filtered_results

def antigener_classification_update_1(img, searcher, id_map, det_threshed=0.2, negative_threshed=0.65, positive_threshed=0.7):
    results = antigener_detector.predict(img, det_threshed)
    results_dict = {'positive': [], 'negative': []}
    for result in results:
        x1, y1, x2, y2 = result['bbox']
        roi_img = img[int(y1):int(y2), int(x1):int(x2)]
        if result['label_name'] == 'positive':
            if result['score'] > positive_threshed:
                results_dict['positive'].append([x1, y1, x2, y2, result['score']])
            else:
                rec_result = rec_predictor.predict(roi_img)
                scores, docs = searcher.search(rec_result, config['IndexProcess']['return_k'])
                if scores[0][0] >= config["IndexProcess"]["score_thres"]:
                    results_dict[id_map[docs[0][0]].split()[1]].append([x1, y1, x2, y2, scores[0][0]])
        else:
            if result['score'] > negative_threshed:
                results_dict['negative'].append([x1, y1, x2, y2, result['score']])
            else:
                rec_result = rec_predictor.predict(roi_img)
                scores, docs = searcher.search(rec_result, config['IndexProcess']['return_k'])
                if scores[0][0] >= config["IndexProcess"]["score_thres"]:
                    results_dict[id_map[docs[0][0]].split()[1]].append([x1, y1, x2, y2, scores[0][0]])
    return results_dict


if __name__ == "__main__":
    img = cv2.imread('D:\\AI Projects\\Shanghai2022\\AntigenClassifier\\tests\\test_images\\FDLOw1RXoAY6Aat.jpg')
    searcher, id_map = Searcher()
    results = antigener_classification_update_1(img, searcher, id_map)
    for result in results['positive']:
        x1, y1, x2, y2, c = result
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        print(result)
    for result in results['negative']:
        x1, y1, x2, y2, c = result
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
        print(result)
    cv2.namedWindow('img', 0)
    cv2.imshow('img', img)
    cv2.waitKey(0)


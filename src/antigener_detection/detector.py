import time
import numpy as np
from predictor import Predictor
from detpreprocess import det_preprocess
from functools import reduce
from preprocess import create_operators

class DetPredictor(Predictor):
    def __init__(self, inference_model_dir, config):
        super().__init__(inference_model_dir)
        self.preprocess_ops = create_operators(config["DetPreProcess"]["transform_ops"])
        self.config = config

    def preprocess(self, img):
        im_info = {
            'scale_factor': np.array(
                [1., 1.], dtype=np.float32),
            'im_shape': np.array(
                img.shape[:2], dtype=np.float32),
            'input_shape': [3, 640, 640],
            "scale_factor": np.array(
                [1., 1.], dtype=np.float32)
        }
        im, im_info = det_preprocess(img, im_info, self.preprocess_ops)
        inputs = self.create_inputs(im, im_info)
        return inputs

    def create_inputs(self, im, im_info):
        """generate input for different model type
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
            model_arch (str): model type
        Returns:
            inputs (dict): input of model
        """
        inputs = {}
        inputs['image'] = np.array((im, )).astype('float32')
        inputs['im_shape'] = np.array(
            (im_info['im_shape'], )).astype('float32')
        inputs['scale_factor'] = np.array(
            (im_info['scale_factor'], )).astype('float32')

        return inputs

    def parse_det_results(self, pred, threshold, label_list):
        keep_indexes = pred[:, 1].argsort()[::-1]
        results = []
        for idx in keep_indexes:
            single_res = pred[idx]
            class_id = int(single_res[0])
            score = single_res[1]
            bbox = single_res[2:]
            if score < threshold:
                continue
            label_name = label_list[class_id]
            results.append({
                "class_id": class_id,
                "score": score,
                "bbox": bbox,
                "label_name": label_name,
            })
        return results

    def predict(self, image, threshold=0.2):
        '''
        Args:
            image (str/np.ndarray): path of image/ np.ndarray read by cv2
            threshold (float): threshold of predicted box' score
        Returns:
            results (dict): include 'boxes': np.ndarray: shape:[N,6], N: number of box,
                            matix element:[class, score, x_min, y_min, x_max, y_max]
                            MaskRCNN's results include 'masks': np.ndarray:
                            shape: [N, im_h, im_w]
        '''
        inputs = self.preprocess(image)
        np_boxes = None
        input_names = self.paddle_predictor.get_input_names()

        for i in range(len(input_names)):
            input_tensor = self.paddle_predictor.get_input_handle(input_names[
                i])
            input_tensor.copy_from_cpu(inputs[input_names[i]])

        t1 = time.time()
        self.paddle_predictor.run()
        output_names = self.paddle_predictor.get_output_names()
        boxes_tensor = self.paddle_predictor.get_output_handle(output_names[0])
        np_boxes = boxes_tensor.copy_to_cpu()
        t2 = time.time()

        #print("Inference: {} ms per batch image".format((t2 - t1) * 1000.0))

        # do not perform postprocess in benchmark mode
        results = []
        if reduce(lambda x, y: x * y, np_boxes.shape) < 6:
            print('[WARNNING] No object detected.')
            results = np.array([])
        else:
            results = np_boxes

        results = self.parse_det_results(results,
                                         self.config["draw_threshold"],
                                         self.config["label_list"])
        return results
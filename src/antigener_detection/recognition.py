import numpy as np
from .predictor import Predictor
from .preprocess import create_operators
from .postprocess import build_postprocess

class RecPredictor(Predictor):
    def __init__(self, config):
        super().__init__(config["Global"]["rec_inference_model_dir"])
        self.preprocess_ops = create_operators(config["RecPreProcess"]["transform_ops"])
        self.postprocess = build_postprocess(config["RecPostProcess"])

    def predict(self, images, feature_normalize=True):
        input_names = self.paddle_predictor.get_input_names()
        input_tensor = self.paddle_predictor.get_input_handle(input_names[0])

        output_names = self.paddle_predictor.get_output_names()
        output_tensor = self.paddle_predictor.get_output_handle(output_names[
            0])

        if not isinstance(images, (list, )):
            images = [images]
        for idx in range(len(images)):
            for ops in self.preprocess_ops:
                images[idx] = ops(images[idx])
        image = np.array(images)

        input_tensor.copy_from_cpu(image)
        self.paddle_predictor.run()
        batch_output = output_tensor.copy_to_cpu()

        if feature_normalize:
            feas_norm = np.sqrt(
                np.sum(np.square(batch_output), axis=1, keepdims=True))
            batch_output = np.divide(batch_output, feas_norm)

        if self.postprocess is not None:
            batch_output = self.postprocess(batch_output)
        return batch_output
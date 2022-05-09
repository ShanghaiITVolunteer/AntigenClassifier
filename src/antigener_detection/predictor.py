import os
from paddle.inference import Config
from paddle.inference import create_predictor

class Predictor(object):
    def __init__(self, inference_model_dir,
                 use_gpu=True, ir_optim=True, gpu_mem=200, cpu_num_threads=4):

        self.paddle_predictor, self.config = self.create_paddle_predictor(inference_model_dir,
                                                                          use_gpu, ir_optim,
                                                                          gpu_mem, cpu_num_threads)

    def predict(self, image):
        raise NotImplementedError

    def create_paddle_predictor(self, inference_model_dir,
                                use_gpu=True, ir_optim=True, gpu_mem=200, cpu_num_threads=4):
        params_file = os.path.join(inference_model_dir, "model.pdiparams")
        model_file = os.path.join(inference_model_dir, "model.pdmodel")
        config = Config(model_file, params_file)

        if use_gpu:
            config.enable_use_gpu(gpu_mem, 0)
        else:
            config.disable_gpu()
        config.set_cpu_math_library_num_threads(cpu_num_threads)

        config.disable_glog_info()
        config.switch_ir_optim(ir_optim)  # default true

        config.enable_memory_optim()
        # use zero copy
        config.switch_use_feed_fetch_ops(False)
        predictor = create_predictor(config)

        return predictor, config
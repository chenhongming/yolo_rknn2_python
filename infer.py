import os
import time

import cv2
import numpy as np

BOX_THRESH = 0.5
NMS_THRESH = 0.5
QUANTIZE_ON = True

data_set = '/data/database/public/calibration_data/coco_person_subset_1000.txt'
# data_set = '/data/database/private/calibration_data/adas_1000.txt'
# data_set = '/home/database/private/calibration_data/night_images2_subset_1000.txt'


# CLASSES = ("person",)


CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
           'cell phone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
           'hair drier', 'toothbrush')


class Detection:

    def __init__(self, rknn_model: str, ir_model: str, model_type: str, img_dir: str, save_dir: str, model_input_shape:list, verbose: bool):
        self.rknn_model = rknn_model
        self.ir_model = ir_model
        self.model_type = model_type
        self.img_dir = img_dir
        self.save_dir = save_dir
        self.verbose = verbose
        self.imgs_list = os.listdir(self.img_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        self.model_input_shape = model_input_shape

        if model_type == 'yolov5':
            # anchor based
            values = [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0,
                      45.0, 59.0, 119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0]
            self.anchors = np.array(values).reshape(3, -1, 2).tolist()
        elif model_type == 'yolov7':
            # anchor based
            values = [12.0, 16.0, 19.0, 36.0, 40.0, 28.0, 36.0, 75.0, 76.0,
                      55.0, 72.0, 146.0, 142.0, 110.0, 192.0, 243.0, 459.0, 401.0]
            self.anchors = np.array(values).reshape(3, -1, 2).tolist()
        elif model_type in ['yolov6', 'yolov8', 'ppyoloe_plus']:
            # anchor free
            # dummy anchors for consociate
            self.anchors = [[[1.0, 1.0]]] * 3

    def test_rknn_simulator_forward(self, log_flag=False, time_elapsed_flag=False):
        # Create RKNN object
        from rknn.api import RKNN
        rknn = RKNN(verbose=self.verbose)

        # pre-process config
        print('--> Config model')
        rknn.config(mean_values=[[0, 0, 0]],
                    std_values=[[255, 255, 255]],
                    quant_img_RGB2BGR=False,
                    quantized_dtype='asymmetric_quantized-8',  # asymmetric_quantized-8(default)、asymmetric_quantized-16
                    quantized_algorithm='normal',  # normal(default), mmse, kl_divergence
                    quantized_method='channel',  # layer, channel(default)
                    float_dtype='float16',  # float16(default), in  non-quantized case
                    optimization_level=2,  # 0, 1, 2, 3(highest)
                    target_platform='rk3588',
                    # custom_stirng=None,  # Add custom string information to RKNN model
                    remove_weight=False,  # Remove the weights to generate a RKNN slave model
                    compress_weight=False,  # Compress the weights of the model
                    single_core_mode=False,  # Whether to generate only single-core model(only for RK3588)
                    model_pruning=True,  # Pruning the model
                    op_target=None,  # Used to specify the target of each operation (NPU/CPU/GPU etc.), the format is
                    # {'op0_output_name':'cpu', 'op1_output_name':'cpu', ...} The currently available
                    # options are: 'cpu' / 'npu' / 'gpu' / 'auto'
                    dynamic_input=None,  # experimental
                    )

        # Load ONNX model
        print('--> Loading model')

        if self.ir_model.endswith('.onnx'):
            ret = rknn.load_onnx(model=self.ir_model,
                                 inputs=None,  # The input node (operand name) of model, input with multiple nodes is
                                 # supported now. All the input node string are placed in a list. The default value is
                                 # None, means get from model
                                 input_size_list=None,  # The shapes of input node, all the input shape are placed in a
                                 # list. If inputs set, the input_size_list should be set also. defualt is None.
                                 input_initial_val=None,  # Set the initial value of the model input, the format is
                                 # ndarray list.The default value is None Mainly used to fix some input as constant,
                                 # For the input that does not need to be fix as a constant, it can be set to None,
                                 # such as [None, np.array([1])]
                                 outputs=None,  # The output node (operand name) of model, output with multiple nodes is
                                 # supported now. All the output nodes are placed in a list. The default value is None,
                                 # means get from model
                                 )
        elif self.ir_model.endswith('.torchscript'):
            ret = rknn.load_pytorch(model=self.ir_model,
                                    input_size_list=[self.model_input_shape])
        else:
            ret = 1
        if ret != 0:
            print('Load model failed!')
            exit(ret)
        print('done')

        # Build model
        print('--> Building model')
        ret = rknn.build(do_quantization=QUANTIZE_ON,
                         dataset=data_set,
                         rknn_batch_size=None,  # Use to adjust batch size of input. default is None (page 29)
                         )
        if ret != 0:
            print('Build model failed!')
            exit(ret)
        print('done')

        # init runtime environment
        print('--> Init runtime environment')
        ret = rknn.init_runtime(target=None,  # Target hardware platform (page 31)
                                device_id=None,  # Device identity number (page 31)
                                perf_debug=False,
                                eval_mem=False,
                                async_mode=False,
                                core_mask=RKNN.NPU_CORE_AUTO,  # only for rk3588 (page 31)
                                )
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)

        # get sdk_version
        # sdk_version = rknn.get_sdk_version()
        # print(sdk_version)

        if time_elapsed_flag:
            img = cv2.imread(self.img_dir + self.imgs_list[0])
            img, _, _ = self.letterbox(img, new_shape=(self.model_input_shape[2], self.model_input_shape[3]))

            # Inference
            print('--> Running model')
            print('test elapsed...')
            st = time.time()
            for _ in range(1000):
                rknn.inference(inputs=[img[None]],
                               data_format=None,  # "nchw" or "nhwc" , only valid for 4-dims input. The default value
                               # is None, means all inputs layout is "nhwc"
                               inputs_pass_through=None,  # (page 32)
                               )
            elapsed = '%.2f' % (time.time() - st)
            # official api
            rknn.eval_perf(inputs=[img[None]], is_print=True)
            print(f'single batchsize rknn elapsed time:{elapsed} ms')
        else:
            for img_name in self.imgs_list:
                img = cv2.imread(self.img_dir + img_name)
                print(f'{self.img_dir + img_name}: |'
                      f'(image shape[h,w]) {img.shape[:2]} | '
                      f'(model shape[h,w]) {self.model_input_shape[2:]} |')
                img_copy = img.copy()
                img, _, _ = self.letterbox(img, new_shape=(self.model_input_shape[2], self.model_input_shape[3]))

                # Inference
                print('--> Running model')
                outputs = rknn.inference(inputs=[img[None]],
                                         data_format=None,
                                         # "nchw" or "nhwc" , only valid for 4-dims input. The default value
                                         # is None, means all inputs layout is "nhwc"
                                         inputs_pass_through=None,  # (page 32)
                                         )
                # print(outputs.shape)

                # post process
                boxes, classes, scores = self.post_process(outputs, anchors=self.anchors, model_type=self.model_type)
                if boxes is not None:
                    boxes = self.scale_boxes(self.model_input_shape[2:], boxes, img_copy.shape[:2])
                    if log_flag:
                        with open(self.save_dir + img_name.split('.')[0] + '_rknn_log.txt', 'w') as log_file:
                            for box, score, cl in zip(boxes, scores, classes):
                                top, left, right, bottom = box
                                text = str(top) + ' ' + str(left) + ' ' + str(right) + ' ' + str(bottom) + ' ' + str(
                                    score) + ' ' + str(cl) + '\n'
                                log_file.writelines(text)
                    self.draw(img_copy, boxes, scores, classes)
                    # cv2.imshow("post process result", img)
                    # cv2.waitKey(1000)
                    cv2.imwrite(self.save_dir + img_name.replace('.', '_rknn.'), img_copy)
                    # cv2.waitKeyEx(0)

        rknn.release()

    def test_rknn_rk3588_forward(self, log_flag=False, time_elapsed_flag=False):
        # Create RKNN object
        from rknnlite.api import RKNNLite
        rknn_lite = RKNNLite(verbose=self.verbose)

        # Load RKNN model
        ret = rknn_lite.load_rknn(self.rknn_model)
        if ret != 0:
            print('Load model failed!')
            exit(ret)

        # Init runtime environment
        print('--> Init runtime environment')
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        if ret != 0:
            print('Init runtime environment failed!')
            exit(ret)
        print('done')

        # get sdk_version
        sdk_version = rknn_lite.get_sdk_version()
        print(sdk_version)

        # self.model_input_shape from 'test_onnx_forward' function
        if time_elapsed_flag:
            img = cv2.imread(self.img_dir + self.imgs_list[0])
            img, _, _ = self.letterbox(img, new_shape=(self.model_input_shape[2], self.model_input_shape[3]))

            # Inference
            print('--> Running model')
            print('test elapsed...')
            st = time.time()
            for _ in range(1000):
                rknn_lite.inference(inputs=[img[None]],
                                    data_format=None  # "nchw" or "nhwc" , only valid for 4-dims input. The default
                                    # value is None, means all inputs layout is "nhwc"
                                    )
            elapsed = '%.2f' % (time.time() - st)
            print(f'single batchsize rknn elapsed time:{elapsed} ms')
        else:
            for img_name in self.imgs_list:
                img = cv2.imread(self.img_dir + img_name)
                print(f'{self.img_dir + img_name}: |'
                      f'(image shape[h,w]) {img.shape[:2]} | '
                      f'(model shape[h, w]) {self.model_input_shape[2:]} |')
                img_copy = img.copy()
                img, _, _ = self.letterbox(img, new_shape=(self.model_input_shape[2], self.model_input_shape[3]))

                # Inference
                print('--> Running model')
                outputs = rknn_lite.inference(inputs=[img[None]],
                                              data_format=None
                                              # "nchw" or "nhwc" , only valid for 4-dims input. The default
                                              # value is None, means all inputs layout is "nhwc"
                                              )
                # print(outputs.shape)

                # post process
                boxes, classes, scores = self.post_process(outputs, anchors=self.anchors, model_type=self.model_type)
                if boxes is not None:
                    boxes = self.scale_boxes(self.model_input_shape[2:], boxes, img_copy.shape[:2])
                    if log_flag:
                        with open(self.save_dir + img_name.split('.')[0] + '_rknn_log.txt', 'w') as log_file:
                            for box, score, cl in zip(boxes, scores, classes):
                                top, left, right, bottom = box
                                text = str(top) + ' ' + str(left) + ' ' + str(right) + ' ' + str(bottom) + ' ' + str(
                                    score) + ' ' + str(cl) + '\n'
                                log_file.writelines(text)
                    self.draw(img_copy, boxes, scores, classes)
                    # cv2.imshow("post process result", img)
                    # cv2.waitKey(1000)
                    cv2.imwrite(self.save_dir + img_name.replace('.', '_rknn.'), img_copy)
                    # cv2.waitKeyEx(0)

        rknn_lite.release()

    def test_ir_forward(self,):
        if self.ir_model.endswith('.onnx'):
            import onnxruntime as rt
            ort_sess = rt.InferenceSession(self.ir_model)
            self.model_input_shape = ort_sess._inputs_meta[0].shape  # (n, c, h, w)

            for img_name in self.imgs_list:
                img = cv2.imread(self.img_dir + img_name)
                print(f'{self.img_dir + img_name}: |'
                      f'(image shape[h,w]) {img.shape[:2]} | '
                      f'(model shape[h, w]) {self.model_input_shape[2:]} |')
                img_copy = img.copy()
                img, _, _ = self.letterbox(img, new_shape=(self.model_input_shape[2], self.model_input_shape[3]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                img = img.astype(np.float32) / 255.
                img = img.transpose((2, 0, 1))
                img = img.reshape((1, *img.shape))

                outputs = ort_sess.run(None, {'images': img})

                # post process
                boxes, classes, scores = self.post_process(outputs, anchors=self.anchors, model_type=self.model_type)
                if boxes is not None:
                    boxes = self.scale_boxes(self.model_input_shape[2:], boxes, img_copy.shape[:2])
                    self.draw(img_copy, boxes, scores, classes)
                    # cv2.imshow("post process result", img_copy)
                    # cv2.waitKey(1000)
                cv2.imwrite(self.save_dir + img_name.replace('.', '_onnx.'), img_copy)
        else:
            print('Not supported ir format.')
            return

    def post_process(self, input_data, anchors, model_type):
        boxes, scores, classes_conf = [], [], []
        if model_type in ['yolov5', 'yolov7']:
            # 1*255*h*w -> 3*85*h*w
            input_data = [_in.reshape([len(anchors[0]), -1] + list(_in.shape[-2:])) for _in in input_data]
            for i in range(len(input_data)):
                boxes.append(self.box_process(input_data[i][:, :4, :, :], anchors[i], model_type))
                scores.append(input_data[i][:, 4:5, :, :])
                classes_conf.append(input_data[i][:, 5:, :, :])
        elif model_type in ['yolov6', 'yolov8', 'ppyoloe_plus']:
            defualt_branch = 3
            pair_per_branch = len(input_data) // defualt_branch
            # Python 忽略 score_sum 输出
            for i in range(defualt_branch):
                boxes.append(self.box_process(input_data[pair_per_branch * i], None, model_type))
                classes_conf.append(input_data[pair_per_branch * i + 1])
                scores.append(np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))

        def sp_flatten(_in):
            ch = _in.shape[1]
            _in = _in.transpose(0, 2, 3, 1)
            return _in.reshape(-1, ch)

        boxes = [sp_flatten(_v) for _v in boxes]
        classes_conf = [sp_flatten(_v) for _v in classes_conf]
        scores = [sp_flatten(_v) for _v in scores]

        boxes = np.concatenate(boxes)
        classes_conf = np.concatenate(classes_conf)
        scores = np.concatenate(scores)

        # filter according to threshold
        boxes, classes, scores = self.filter_boxes(boxes, scores, classes_conf, model_type)

        # nms
        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]
            keep = self.nms_boxes(b, s)

            if len(keep) != 0:
                nboxes.append(b[keep])
                nclasses.append(c[keep])
                nscores.append(s[keep])

        if not nclasses and not nscores:
            return None, None, None

        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)

        return boxes, classes, scores

    def box_process(self, position, anchors, model_type):
        grid_h, grid_w = position.shape[2:4]
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        col = col.reshape(1, 1, grid_h, grid_w)
        row = row.reshape(1, 1, grid_h, grid_w)
        grid = np.concatenate((col, row), axis=1)
        im_size = self.model_input_shape[2:]
        stride = np.array([im_size[0] // grid_h, im_size[1] // grid_w]).reshape(1, 2, 1, 1)

        if model_type in ['yolov5', 'yolov7', 'yolox']:
            # output format: xywh -> xyxy
            if model_type == 'yolox':
                box_xy = position[:, :2, :, :]
                box_wh = np.exp(position[:, 2:4, :, :]) * stride
            elif model_type in ['yolov5', 'yolov7']:
                col = col.repeat(len(anchors), axis=0)
                row = row.repeat(len(anchors), axis=0)
                anchors = np.array(anchors)
                anchors = anchors.reshape(*anchors.shape, 1, 1)

                box_xy = position[:, :2, :, :] * 2 - 0.5
                box_wh = pow(position[:, 2:4, :, :] * 2, 2) * anchors

            box_xy += grid
            box_xy *= stride
            box = np.concatenate((box_xy, box_wh), axis=1)

            # Convert [c_x, c_y, w, h] to [x1, y1, x2, y2]
            xyxy = np.copy(box)
            xyxy[:, 0, :, :] = box[:, 0, :, :] - box[:, 2, :, :] / 2  # top left x
            xyxy[:, 1, :, :] = box[:, 1, :, :] - box[:, 3, :, :] / 2  # top left y
            xyxy[:, 2, :, :] = box[:, 0, :, :] + box[:, 2, :, :] / 2  # bottom right x
            xyxy[:, 3, :, :] = box[:, 1, :, :] + box[:, 3, :, :] / 2  # bottom right y

        elif model_type == 'yolov6' and position.shape[1] == 4:
            box_xy = grid + 0.5 - position[:, 0:2, :, :]
            box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
            xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

        elif model_type in ['yolov6', 'yolov8', 'ppyoloe_plus']:
            position = self.dfl(position)
            box_xy = grid + 0.5 - position[:, 0:2, :, :]
            box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
            xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

        return xyxy

    def filter_boxes(self, boxes, box_confidences, box_class_probs, model_type):
        """Filter boxes with object threshold.
        """
        box_confidences = box_confidences.reshape(-1)
        candidate, class_num = box_class_probs.shape

        class_max_score = np.max(box_class_probs, axis=-1)
        classes = np.argmax(box_class_probs, axis=-1)

        if model_type == 'yolov7' and class_num == 1:
            _class_pos = np.where(box_confidences >= BOX_THRESH)
            scores = box_confidences[_class_pos]
        else:
            _class_pos = np.where(class_max_score * box_confidences >= BOX_THRESH)
            scores = (class_max_score * box_confidences)[_class_pos]

        boxes = boxes[_class_pos]
        classes = classes[_class_pos]

        return boxes, classes, scores

    def nms_boxes(self, boxes, scores):
        """Suppress non-maximal boxes.
        # Returns
            keep: ndarray, index of effective boxes.
        """
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
            h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= NMS_THRESH)[0]
            order = order[inds + 1]
        keep = np.array(keep)
        return keep

    def process(self, input, mask, anchors):

        anchors = [anchors[i] for i in mask]
        grid_h, grid_w = map(int, input.shape[0:2])

        box_confidence = self.sigmoid(input[..., 4])
        box_confidence = np.expand_dims(box_confidence, axis=-1)

        box_class_probs = self.sigmoid(input[..., 5:])

        box_xy = self.sigmoid(input[..., :2]) * 2 - 0.5
        # print(box_xy)
        col = np.tile(np.arange(0, grid_w), grid_h).reshape(-1, grid_w)
        row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_w)
        # print("row:", row.shape)
        col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        grid = np.concatenate((col, row), axis=-1)
        box_xy += grid
        box_xy *= (int(self.model_input_shape[2] / grid_h), int(self.model_input_shape[3] / grid_w))
        # print(box_xy.shape)

        box_wh = pow(self.sigmoid(input[..., 2:4]) * 2, 2)
        box_wh = box_wh * anchors

        box = np.concatenate((box_xy, box_wh), axis=-1)
        # print(box_confidence.shape)

        return box, box_confidence, box_class_probs

    def scale_boxes(self, img1_shape, boxes, img0_shape, ratio_pad=None):
        # Rescale boxes (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        boxes[..., [0, 2]] -= pad[0]  # x padding
        boxes[..., [1, 3]] -= pad[1]  # y padding
        boxes[..., :4] /= gain
        self.clip_boxes(boxes, img0_shape)
        return boxes

    @staticmethod
    def dfl(position):
        # Distribution Focal Loss (DFL)
        import torch
        x = torch.tensor(position)
        n, c, h, w = x.shape
        p_num = 4
        mc = c // p_num
        y = x.reshape(n, p_num, mc, h, w)
        y = y.softmax(2)
        acc_metrix = torch.tensor(range(mc)).float().reshape(1, 1, mc, 1, 1)
        y = (y * acc_metrix).sum(2)
        return y.numpy()

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def xywh2xyxy(x):
        # Convert [x, y, w, h] to [x1, y1, x2, y2]
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    @staticmethod
    def draw(image, boxes, scores, classes):
        """Draw the boxes on the image.

        # Argument:
            image: original image.
            boxes: ndarray, boxes of objects.
            classes: ndarray, classes of objects.
            scores: ndarray, scores of objects.
            all_classes: all classes name.
        """
        for box, score, cl in zip(boxes, scores, classes):
            top, left, right, bottom = box
            # print('class: {}, score: {}'.format(CLASSES[cl], score))
            # print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
            top = int(top)
            left = int(left)
            right = int(right)
            bottom = int(bottom)

            cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
            cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                        (top, left + 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)

    @staticmethod
    def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)

    @staticmethod
    def clip_boxes(boxes, shape):
        # Clip boxes (xyxy) to image shape (height, width)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def main():
    rknn_model = './target_weights/yolov8s_384_640_1.rknn'
    ir_model = './ir_weights/yolov8s_384_640_1.onnx'
    model_type = 'yolov8'
    model_input_shape = [1, 3, 384, 640]

    img_dir = './images/'
    save_dir = 'results/'
    verbose = False

    detect = Detection(rknn_model=rknn_model,
                       ir_model=ir_model,
                       model_type=model_type,
                       img_dir=img_dir,
                       save_dir=save_dir,
                       model_input_shape=model_input_shape,
                       verbose=verbose)

    # print('------- ir forward start -------')
    # detect.test_ir_forward()
    # print('------- ir forward end -------', '\n')

    # print('------- rknn forward start -------')
    # detect.test_rknn_simulator_forward(time_elapsed_flag=False)
    # print('------- rknn forward end-------')

    print('------- rknn forward start -------')
    detect.test_rknn_rk3588_forward(time_elapsed_flag=False)
    print('------- rknn forward end-------')


if __name__ == '__main__':
    main()

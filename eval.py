import os
import cv2
import json
import torch
import numpy as np
import onnxruntime as rt

from tqdm import tqdm
from pathlib import Path

from utils import metrics, general

IMG_FORMATS = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]

BOX_THRESH = 0.001
NMS_THRESH = 0.6
QUANTIZE_ON = True

data_set = '/data/database/public/calibration_data/coco_person_subset_1000.txt'
# data_set = '/data/database/private/calibration_data/adas_1000.txt'
# data_set = '/home/database/private/calibration_data/night_images2_subset_1000.txt'


CLASSES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
           8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter',
           13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',
           21: 'bear',
           22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
           27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
           34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard',
           38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
           45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
           50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch',
           58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
           62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
           69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
           74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}


# CLASSES = {0: 'person', }


class Detection:

    def __init__(self, rknn_model: str, ir_model: str, model_type: str, img_dir: str, save_dir: str,
                 model_input_shape: list, verbose: bool):
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

    def val_rknn_rk3588_forward(self, save_json=False, save_dir='result_json/', anno_json=''):
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

        # init prarm
        nc = len(CLASSES)
        iouv = torch.linspace(0.5, 0.95, 10, device='cpu')  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()
        seen = 0
        class_map = list(range(1000))
        s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
        dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        jdict, stats, ap, ap_class = [], [], [], []

        pbar = tqdm(self.imgs_list, desc=s, bar_format=general.TQDM_BAR_FORMAT)  # progress bar
        for img_name in pbar:
            # load img
            img = cv2.imread(self.img_dir + img_name)
            img_copy = img.copy()
            img, ratio, pad = self.letterbox(img, new_shape=(self.model_input_shape[2], self.model_input_shape[3]))

            # load labels
            h, w = img_copy.shape[:2]
            label_path = self.img_dir.replace('images', 'labels') + img_name.split('.')[0] + '.txt'
            if not os.path.isfile(label_path):
                print('invalid label file: ', label_path)
                continue
            label = self.load_label(label_path)
            if label.size:  # normalized xywh to pixel xyxy format
                label[:, 1:] = general.xywhn2xyxy(label[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            nl = len(label)  # number of labels
            if nl:
                label[:, 1:5] = general.xyxy2xywhn(label[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)
                targets = torch.zeros((nl, 6))
                targets[:, 1:] = torch.from_numpy(label)
                targets[:, 2:] *= torch.tensor((img.shape[1], img.shape[0], img.shape[1], img.shape[0]),
                                               device='cpu')  # to pixels

            # Inference
            print('--> Running model')
            outputs = rknn_lite.inference(inputs=[img[None]],
                                          data_format=None
                                          # "nchw" or "nhwc" , only valid for 4-dims input. The default
                                          # value is None, means all inputs layout is "nhwc"
                                          )
            # post process
            boxes, classes, scores = self.post_process(outputs, anchors=self.anchors, model_type=self.model_type)
            out = [torch.zeros((0, 6))] * 1
            if boxes is not None:
                classes = classes.reshape(-1, 1)
                scores = scores.reshape(-1, 1)
                out_tmp = np.hstack((boxes, scores, classes))
                out_tmp = np.array(out_tmp, dtype=np.float32)
                out_tmp = torch.from_numpy(out_tmp)
                out[0] = out_tmp

            # Metrics
            for si, pred in enumerate(out):
                labels = targets[targets[:, 0] == si, 1:]
                path = Path(self.img_dir + img_name)
                nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
                correct = torch.zeros(npr, niou, dtype=torch.bool, device='cpu')  # init
                seen += 1
                predn = pred.clone()
                predn[:, :4] = self.scale_boxes(self.model_input_shape[2:], predn[:, :4], img_copy.shape[:2])

                # Evaluate
                if nl:
                    tbox = general.xywh2xyxy(labels[:, 1:5])  # target boxes

                    # native-space labels
                    tbox[:, :4] = self.scale_boxes(self.model_input_shape[2:], tbox[:, :4], img_copy.shape[:2])
                    labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                    correct = metrics.process_batch(predn, labelsn, iouv)
                stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)
                if save_json:
                    self.save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary

        # Compute metrics
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy

        if len(stats) and stats[0].any():
            tp, fp, p, r, f1, ap, ap_class = metrics.ap_per_class(*stats)

            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

        # Print results
        pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
        print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
        if nt.sum() == 0:
            print(f'WARNING: no labels found in val set, can not compute metrics without labels ⚠️')

        # Print results per class
        if nc > 1 and len(stats):
            for i, c in enumerate(ap_class):
                print(pf % (CLASSES[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

        # Save JSON
        if save_json and len(jdict):
            os.makedirs(save_dir, exist_ok=True)
            w = Path(self.ir_model).stem if self.ir_model is not None else ''  # weights
            if not anno_json.endswith('.json'):  # annotations json
                print('invalid json file')
                exit()  # annotations json
            pred_json = save_dir + f"{w}_predictions.json"  # predictions json
            print(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
            with open(pred_json, 'w') as f:
                json.dump(jdict, f)

            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                general.check_requirements(['pycocotools'])
                from pycocotools.coco import COCO
                from pycocotools.cocoeval import COCOeval

                anno = COCO(anno_json)  # init annotations api
                pred = anno.loadRes(pred_json)  # init predictions api
                eval = COCOeval(anno, pred, 'bbox')
                eval.evaluate()
                eval.accumulate()
                eval.summarize()
            except Exception as e:
                print(f'pycocotools unable to run: {e}')
        rknn_lite.release()

    def val_ir_forward(self, save_json=False, save_dir='result_json/', anno_json=''):
        if self.ir_model.endswith('.onnx'):
            import onnxruntime as rt
            ort_sess = rt.InferenceSession(self.ir_model)
            self.model_input_shape = ort_sess._inputs_meta[0].shape  # (n, c, h, w)

            # init param
            nc = len(CLASSES)
            iouv = torch.linspace(0.5, 0.95, 10, device='cpu')  # iou vector for mAP@0.5:0.95
            niou = iouv.numel()
            seen = 0
            class_map = list(range(1000))
            s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
            dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            jdict, stats, ap, ap_class = [], [], [], []

            pbar = tqdm(self.imgs_list, desc=s, bar_format=general.TQDM_BAR_FORMAT)  # progress bar
            for img_name in pbar:
                # load image
                img = cv2.imread(self.img_dir + img_name)
                img_copy = img.copy()
                img, ratio, pad = self.letterbox(img, new_shape=(self.model_input_shape[2], self.model_input_shape[3]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                img = img.astype(np.float32) / 255.
                img = img.transpose((2, 0, 1))
                img = img.reshape((1, *img.shape))

                # load labels
                h, w = img_copy.shape[:2]
                label_path = self.img_dir.replace('images', 'labels') + img_name.split('.')[0] + '.txt'
                if not os.path.isfile(label_path):
                    print('invalid label file: ', label_path)
                    continue
                label = self.load_label(label_path)
                if label.size:  # normalized xywh to pixel xyxy format
                    label[:, 1:] = general.xywhn2xyxy(label[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0],
                                                      padh=pad[1])

                nl = len(label)  # number of labels
                if nl:
                    label[:, 1:5] = general.xyxy2xywhn(label[:, 1:5], w=img.shape[3], h=img.shape[2], clip=True,
                                                       eps=1E-3)
                    targets = torch.zeros((nl, 6))
                    targets[:, 1:] = torch.from_numpy(label)
                    targets[:, 2:] *= torch.tensor((img.shape[3], img.shape[2], img.shape[3], img.shape[2]),
                                                   device='cpu')  # to pixels

                # forward
                outputs = ort_sess.run(None, {'images': img})

                # post process
                boxes, classes, scores = self.post_process(outputs, anchors=self.anchors, model_type=self.model_type)
                out = [torch.zeros((0, 6))] * 1
                if boxes is not None:
                    classes = classes.reshape(-1, 1)
                    scores = scores.reshape(-1, 1)
                    out_tmp = np.hstack((boxes, scores, classes))
                    out_tmp = np.array(out_tmp, dtype=np.float32)
                    out_tmp = torch.from_numpy(out_tmp)
                    out[0] = out_tmp
                # Metrics
                for si, pred in enumerate(out):
                    labels = targets[targets[:, 0] == si, 1:]
                    path = Path(self.img_dir + img_name)
                    nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
                    correct = torch.zeros(npr, niou, dtype=torch.bool, device='cpu')  # init
                    seen += 1
                    predn = pred.clone()
                    predn[:, :4] = self.scale_boxes(self.model_input_shape[2:], predn[:, :4], img_copy.shape[:2])

                    # Evaluate
                    if nl:
                        tbox = general.xywh2xyxy(labels[:, 1:5])  # target boxes

                        # native-space labels
                        tbox[:, :4] = self.scale_boxes(self.model_input_shape[2:], tbox[:, :4], img_copy.shape[:2])
                        labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                        correct = metrics.process_batch(predn, labelsn, iouv)
                    stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)
                    if save_json:
                        self.save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            # Compute metrics
            stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy

            if len(stats) and stats[0].any():
                tp, fp, p, r, f1, ap, ap_class = metrics.ap_per_class(*stats)

                ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
                mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

            # Print results
            pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
            print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
            if nt.sum() == 0:
                print(f'WARNING: no labels found in val set, can not compute metrics without labels ⚠️')

            # Print results per class
            if nc > 1 and len(stats):
                for i, c in enumerate(ap_class):
                    print(pf % (CLASSES[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

            # Save JSON
            if save_json and len(jdict):
                os.makedirs(save_dir, exist_ok=True)
                w = Path(self.ir_model).stem if self.ir_model is not None else ''  # weights
                if not anno_json.endswith('.json'):  # annotations json
                    print('invalid json file')
                    exit()
                pred_json = save_dir + f"{w}_predictions.json"  # predictions json
                print(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
                with open(pred_json, 'w') as f:
                    json.dump(jdict, f)

                try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                    general.check_requirements(['pycocotools'])
                    from pycocotools.coco import COCO
                    from pycocotools.cocoeval import COCOeval

                    anno = COCO(anno_json)  # init annotations api
                    pred = anno.loadRes(pred_json)  # init predictions api
                    eval = COCOeval(anno, pred, 'bbox')
                    eval.evaluate()
                    eval.accumulate()
                    eval.summarize()
                except Exception as e:
                    print(f'pycocotools unable to run: {e}')
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

    @staticmethod
    def load_label(path):
        with open(path) as f:
            # label
            lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)
        return lb

    @staticmethod
    def save_one_json(predn, jdict, path, class_map):
        # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
        image_id = int(path.stem) if path.stem.isnumeric() else path.stem
        box = general.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for p, b in zip(predn.tolist(), box.tolist()):
            jdict.append({
                'image_id': image_id,
                'category_id': class_map[int(p[5])],
                'bbox': [round(x, 3) for x in b],
                'score': round(p[4], 5)})


def main():
    rknn_model = './target_weights/yolov6s_384_640_1.rknn'
    ir_model = './ir_weights/yolov6s_384_640_1.onnx'
    model_type = 'yolov6'
    model_input_shape = [1, 3, 384, 640]

    img_dir = '/media/smdt/B626954326950591/database/private/coco_val/images/val2017/'
    save_dir = 'results/'
    anno_json = '/media/smdt/B626954326950591/database/private/coco_val/annotations/instances_val2017.json'
    verbose = False

    detect = Detection(rknn_model=rknn_model,
                       ir_model=ir_model,
                       model_type=model_type,
                       img_dir=img_dir,
                       save_dir=save_dir,
                       model_input_shape=model_input_shape,
                       verbose=verbose)

    print('\t\t\t\t------- onnx forward start -------')
    detect.val_ir_forward(save_json=True, save_dir=save_dir, anno_json=anno_json)
    print('\t\t\t\t------- onnx forward end -------', '\n')

    print('\t\t\t\t------- rknn forward start -------')
    detect.val_rknn_rk3588_forward(save_json=True, save_dir=save_dir, anno_json=anno_json)
    print('\t\t\t\t------- rknn forward end-------')


if __name__ == '__main__':
    main()

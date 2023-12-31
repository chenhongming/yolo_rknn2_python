"""Exports a YOLOv5 *.pt model to ONNX and TorchScript formats

Usage:
    $ export PYTHONPATH="$PWD" && python cvt/export_yolov5.py --weights ./weights/yolov5s.pt --img 640 --batch 1
"""

import argparse
import sys
import time

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn

import models
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU
from utils.general import set_logging, check_img_size

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='weights path')  # from yolov5/models/
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')

    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)
    set_logging()
    t = time.time()

    # Load PyTorch model
    # model = attempt_load(opt.weights, map_location=torch.device('cpu'))  # load FP32 model
    model = attempt_load(opt.weights)  # load FP32 model

    labels = model.names

    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples

    # Input
    img = torch.zeros(opt.batch_size, 3, *opt.img_size)  # image size(1,3,320,192) iDetection

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                # m.act = SiLU()
                m.act = nn.ReLU()
    model.model[-1].export = True  # set Detect() layer export=True
    y = model(img)  # dry run
    try:
        import onnx

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = opt.weights.replace('.pt', f'_{opt.img_size[0]}_{opt.img_size[1]}_{opt.batch_size}.onnx')  # filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['images'],
                          output_names=['classes', 'boxes'] if y is None else ['output'])
        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        output_node = onnx_model.graph.output[0]
        onnx.checker.check_model(onnx_model)  # check onnx model

        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        # Simplify
        if opt.simplify:
            try:
                import onnxsim

                print(f'simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(onnx_model)
                assert check, 'assert check failed'
                onnx.save(model_onnx, f)
            except Exception as e:
                print(f'simplifier failure: {e}')
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)


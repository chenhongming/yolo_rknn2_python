## 🤗 Introduction

A development tool for converting yolo series models to rknn models.



## 😇 Development Platform

* Host Ubuntu 18.04 x86_64  rknn_toolkit2  (Python)

* rk3588 Ubuntu 20.04 aarch64  rknn_toolkit_lite2  (Python && C++)



## 🥳 Installation

### &ensp; 0. Clone the repository:

```
git clone https://github.com/rockchip-linux/rknn-toolkit2.git
```

### &ensp; 1. For Host Environment:

```
conda create -n rknn2 python=3.6
conda activate rknn2

sudo apt-get install libxslt1-dev zlib1g zlib1g-dev libglib2.0-0 libsm6 libgl1-mesa-glx libprotobuf-dev gcc

cd rknn-toolkit2/
pip install -r doc/requirements_cp36-1.5.2.txt -i https://mirror.baidu.com/pypi/simple

cd packages/
pip install rknn_toolkit2-1.5.2+b642f30c-cp36-cp36m-linux_x86_64.whl
```

```
# check
python3
from rknn.api import RKNN
```



### &ensp; 2. For Device(rk3588) Environment:

```
conda create -n rknn2 python=3.8
conda activate rknn2

sudo apt-get install libxslt1-dev zlib1g zlib1g-dev libglib2.0-0 libsm6 libgl1-mesa-glx libprotobuf-dev gcc

cd rknn-toolkit2/
# tensorflow==2.8.0 (no need install)
pip install -r doc/requirements_cp38-1.5.2.txt -i https://mirror.baidu.com/pypi/simple

cd rknn_toolkit_lite2/packages/
pip install rknn_toolkit_lite2-1.5.2-cp38-cp38-linux_aarch64.whl
```

```
# check
python3
from rknnlite.api import RKNN
```



## 🤭 Supported Algorithms

-  [YOLOv5]() ✅
-  [YOLOv6]() ✅
-  [YOLOv7]() 🔜
-  [YOLOv8]() ✅

## 😜 Get started
### 🛠️ Deploy for yolov5 (version 7.0)

####   1. pytorch2onnx

* Please clone the official repository of [yolov5](https://github.com/ultralytics/yolov5) and install the [dependencies](https://github.com/ultralytics/yolov5/blob/master/requirements.txt).

* Copy the [cvt/export_yolov5.py](https://github.com/chenhongming/yolo_rknn2_python/blob/master/cvt/export_yolov5.py) to the **root** directory of [yolov5](https://github.com/ultralytics/yolov5).

* Replace the official forward function of the Detect class in [models/yolo.py](https://github.com/ultralytics/yolov5/blob/master/models/yolo.py) with the following forward function.

```
# export onnx for rknn (multi head)
def forward(self, x):
    self.training |= self.export
    for i in range(self.nl):
        x[i] = self.m[i](x[i])  # conv
        bs, _, ny, nx = x[i].shape  # x(bs,255,20,20)
        x[i] = x[i].view(bs, self.na * self.no, ny, nx).contiguous()
    return x
```
* run
```
# opset_version=11

python export_onnx.py --weights ./yolov5s.pt --img 640 640 --batch-size 1 --simplify
```
* The onnx model is stored in the same directory as the pytorch model.

* Copy the generated onnx model to [ir_weights](https://github.com/chenhongming/yolo_rknn2_python/tree/master/ir_weights).



#### 2. onnx2rknn

* Generate a calibration dataset from the training or validation set via [utils/get_calibration.py](https://github.com/chenhongming/yolo_rknn2_python/blob/master/utils/get_calibration.py).


```
# params
# im_dir: src im path
# calibration_dir: dst im path

cd utils && python get_calibration.py
```

* Run [export.py](https://github.com/chenhongming/yolo_rknn2_python/blob/master/export.py) to generate the quantized rknn model(uint8).

```
# params
# rknn_batch_size = 1
# IR_MODEL = './ir_weights/yolov5s_384_640_1.onnx'
# RKNN_MODEL = f'./target_weights/yolov5s_384_640_{rknn_batch_size}.rknn'
# data_set = '/data/database/public/calibration_data/calibration.txt'

python export.py
```

### 🛠️ Deploy for yolov6 (version 0.4.1)

#### 1. pytorch2onnx

* Please clone the official repository of [YOLOv6](https://github.com/meituan/YOLOv6) and install the [dependencies](https://github.com/meituan/YOLOv6/blob/main/requirements.txt).

* Modify ./yolov6/models/heads/effidehead_distill_ns.py for v6n or v6s

  ```
  // add follow in 'class Detect(nn.Module)'  *line 78*
  
  def _rknn_opt_head(self, x):
      output_for_rknn = []
      for i in range(self.nl):
          x[i] = self.stems[i](x[i])
          reg_feat = self.reg_convs[i](x[i])
          reg_output = self.reg_preds[i](reg_feat)
  
          cls_feat = self.cls_convs[i](x[i])
          cls_output = self.cls_preds[i](cls_feat)
          cls_output = torch.sigmoid(cls_output)
  
          cls_sum = torch.clamp(cls_output.sum(1, keepdim=True), 0, 1)
  
          output_for_rknn.append(reg_output)
          output_for_rknn.append(cls_output)
          output_for_rknn.append(cls_sum)
      return output_for_rknn
  ```

  ```
  // add follow in ''def forward(self, x)'  *line 80*
  
  def forward(self, x):
      if getattr(self, "export_rknn", False):
          return self._rknn_opt_head(x)
  ```

* Modify yolov6/models/effidehead.py for v6m or v6l

    ```
    // add follow in 'class Detect(nn.Module)'  *line 70*
    
    def _rknn_opt_head(self, x):
        output_for_rknn = []
        for i in range(self.nl):
            x[i] = self.stems[i](x[i])
            reg_feat = self.reg_convs[i](x[i])
            reg_output = self.reg_preds[i](reg_feat)
    
            cls_feat = self.cls_convs[i](x[i])
            cls_output = self.cls_preds[i](cls_feat)
            cls_output = torch.sigmoid(cls_output)
    
            cls_sum = torch.clamp(cls_output.sum(1, keepdim=True), 0, 1)
    
            output_for_rknn.append(reg_output)
            output_for_rknn.append(cls_output)
            output_for_rknn.append(cls_sum)
        return output_for_rknn
    ```

    ```
    // add follow in ''def forward(self, x)'  *line 72*
    
    def forward(self, x):
        if getattr(self, "export_rknn", False):
            return self._rknn_opt_head(x)
    ```

* Modify yolov6/layers/common.py

  ```
  // add follow  *line 24*
  
  class Conv(nn.Module):
      '''Normal Conv with SiLU activation'''
      def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False):
          super().__init__()
          padding = kernel_size // 2
          self.conv = nn.Conv2d(
              in_channels,
              out_channels,
              kernel_size=kernel_size,
              stride=stride,
              padding=padding,
              groups=groups,
              bias=bias,
          )
          self.bn = nn.BatchNorm2d(out_channels)
          self.act = nn.SiLU()
  
      def forward(self, x):
          return self.act(self.bn(self.conv(x)))
  
      def forward_fuse(self, x):
          return self.act(self.conv(x))
  ```

* Copy the [cvt/export_yolov6.py](https://github.com/chenhongming/yolo_rknn2_python/blob/master/cvt/export_yolov6.py) to the **tools** of [yolov6](https://github.com/meituan/YOLOv6).

* python tools/export_yolov6.py --weights yolov6s.pt --img-size 384 640

* The onnx model is stored in the same directory as the pytorch model.

* Copy the generated onnx model to [ir_weights](https://github.com/chenhongming/yolo_rknn2_python/tree/master/ir_weights).

  

#### 2. onnx2rknn

* Generate a calibration dataset from the training or validation set via [utils/get_calibration.py](https://github.com/chenhongming/yolo_rknn2_python/blob/master/utils/get_calibration.py).


```
# params
# im_dir: src im path
# calibration_dir: dst im path

cd utils && python get_calibration.py
```

* Run [export.py](https://github.com/chenhongming/yolo_rknn2_python/blob/master/export.py) to generate the quantized rknn model(uint8).

```
# params
# rknn_batch_size = 1
# IR_MODEL = './ir_weights/yolov6s_384_640_1.onnx'
# RKNN_MODEL = f'./target_weights/yolov6s_384_640_{rknn_batch_size}.rknn'
# data_set = '/data/database/public/calibration_data/calibration.txt'

python export.py
```



### 🛠️ Deploy for yolov8 (version 8.0.220)

#### 1. pytorch2torchscript or pytorch2onnx

* Please clone the official repository of [YOLOv8](https://github.com/ultralytics/ultralytics.git) and install the [dependencies](https://github.com/ultralytics/ultralytics/blob/main/requirements.txt).

* Modify ultralytics/engine/exporter.py

  ```
  add ['RKNN', 'rknn', '_rknn.torchscript', True, False] in 'def export_formats()'  *line 95*
  ```

  ```
  add rknn in 'def __call__(self, model=None)' *line 166*
  ```

  ```
  // add follow in 'def __call__(self, model=None)' *line 274*
  if rknn:
      f[12], _ = self.export_rknn()
  ```
  
* Note  **export torchscript for rknn**
  ```
  // add follow in 'class Exporter' *line 309*
  
  @try_export
  def export_rknn(self, prefix=colorstr('RKNN:')):
      """YOLOv8 RKNN model export."""
      LOGGER.info(f'\n{prefix} starting export with torch {torch.__version__}...')
  	
      # export torchscript for rknn
      ts = torch.jit.trace(self.model, self.im, strict=False)
      f = str(self.file).replace(self.file.suffix, f'_rknn.torchscript')
      torch.jit.save(ts, str(f))
  
      LOGGER.info(f'\n{prefix} feed {f} to RKNN-Toolkit or RKNN-Toolkit2 to generate RKNN model.\n'
                  'Refer https://github.com/airockchip/rknn_model_zoo/tree/main/models/CV/object_detection/yolo')
      return f, None
  ```
  
* Note  **export onnx for rknn**

  ```
  // add follow in 'class Exporter' *line 309*
  
  @try_export
  def export_rknn(self, prefix=colorstr('RKNN:')):
      """YOLOv8 RKNN model export."""
      LOGGER.info(f'\n{prefix} starting export with torch {torch.__version__}...')
  
      # export onnx for rknn
      f = str(self.file).replace(self.file.suffix, f'_rknn.onnx')
      try:
          import onnx
  
          print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
          torch.onnx.export(self.model, self.im, f, verbose=False, opset_version=12, input_names=['images'],
                            output_names=['output'])
          # Checks
          onnx_model = onnx.load(f)  # load onnx model
          onnx.checker.check_model(onnx_model)  # check onnx model
          # Simplify
          try:
              import onnxsim
  
              print(f'simplifying with onnx-simplifier {onnxsim.__version__}...')
              model_onnx, check = onnxsim.simplify(onnx_model)
              assert check, 'assert check failed'
          except Exception as e:
              print(f'simplifier failure: {e}')
          print('ONNX export success, saved as %s' % f)
      except Exception as e:
          print('ONNX export failure: %s' % e)
      return f, None
  ```

  


* Modify ultralytics/nn/modules/head.py

  ```
  // add follow in 'class Detect(nn.Module)'  *line 44*
  
  if self.export and self.format == 'rknn':
      y = []
      for i in range(self.nl):
          y.append(self.cv2[i](x[i]))
          cls = torch.sigmoid(self.cv3[i](x[i]))
          cls_sum = torch.clamp(y[-1].sum(1, keepdim=True), 0, 1)
          y.append(cls)
          y.append(cls_sum)
      return y
  ```

* Copy the [cvt/export_yolov8.py](https://github.com/chenhongming/yolo_rknn2_python/blob/master/cvt/export_yolov8.py) to the **root** of [yolov8](https://github.com/ultralytics/ultralytics.git).

* python export.py

* The torchscript model is stored in the same directory as the pytorch model.

* Copy the generated onnx model to [ir_weights]().

#### 2. torchscript2rknn or onnx2rknn

* Generate a calibration dataset from the training or validation set via [utils/get_calibration.py](https://github.com/chenhongming/yolo_rknn2_python/blob/master/utils/get_calibration.py).


```
# params
# im_dir: src im path
# calibration_dir: dst im path

cd utils && python get_calibration.py
```

* Run [export.py](https://github.com/chenhongming/yolo_rknn2_python/blob/master/export.py) to generate the quantized rknn model(uint8).

```
# params
# rknn_batch_size = 1
# IR_MODEL = './ir_weights/yolov8s_384_640_1.torchscript'
# RKNN_MODEL = f'./target_weights/yolov8s_384_640_{rknn_batch_size}.rknn'
# data_set = '/data/database/public/calibration_data/calibration.txt'

python export.py
```



### &ensp;  🔥 Infer for onnx and rknn (python)

* run
```
python infer.py
```

### &ensp;  🔥 Infer for rknn(C++)

[**yolo_anchor_free(yolov6 && yolov8)**](https://github.com/chenhongming/yolo_anchor_free)

[**yolo_anchor_based(yolov5)**](https://github.com/chenhongming/yolo_anchor_based)


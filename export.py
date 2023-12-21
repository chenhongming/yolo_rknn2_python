from rknn.api import RKNN

QUANTIZE_ON = True
rknn_batch_size = 1

crypt = True
crypt_level = 2  # 1, 2, 3
crypt_type = 1
crypt_suffix = ['rknn', 'bin']

# Model from https://github.com/airockchip/rknn_model_zoo
IR_MODEL = './ir_weights/yolov5s_384_640_1.onnx'
RKNN_MODEL = f'./target_weights/yolov5s_384_640_{rknn_batch_size}.rknn'

# data_set = '/data/database/public/calibration_data/coco_person_subset_1000.txt'
data_set = '/data/chm/database/private/calibration_data/adas_1000_.txt'


if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN(verbose=False)

    # pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]],
                std_values=[[255, 255, 255]],
                quant_img_RGB2BGR=False,
                quantized_dtype='asymmetric_quantized-8',  # asymmetric_quantized-8(default)、asymmetric_quantized-16
                quantized_algorithm='normal',  # normal(default), mmse, kl_divergence
                quantized_method='channel',  # layer, channel(default)
                float_dtype='float16',  # float16(default), in  non-quantized case
                optimization_level=3,  # 0, 1, 2, 3(highest)
                target_platform='rk3588',
                # custom_stirng=None,  # Add custom string information to RKNN model
                remove_weight=False,  # Remove the weights to generate a RKNN slave model
                compress_weight=False,  # Compress the weights of the model
                single_core_mode=False,  # Whether to generate only single-core model(only for RK3588)
                model_pruning=False,  # Pruning the model
                op_target=None,  # Used to specify the target of each operation (NPU/CPU/GPU etc.), the format is
                # {'op0_output_name':'cpu', 'op1_output_name':'cpu', ...} The currently available
                # options are: 'cpu' / 'npu' / 'gpu' / 'auto'
                dynamic_input=None,  # experimental
                )
    print('done')

    # Load ONNX or torchscript model
    print('--> Loading model')
    if IR_MODEL.endswith('.onnx'):
        ret = rknn.load_onnx(model=IR_MODEL)
    elif IR_MODEL.endswith('.torchscript'):
        ret = rknn.load_pytorch(model=IR_MODEL, input_size_list=[[1, 3, 384, 640]])
    else:
        ret = 1
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=data_set, rknn_batch_size=rknn_batch_size)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    if crypt:
        RKNN_MODEL_ENCRYPTED = RKNN_MODEL.replace('.rknn', f"_encrypt.{crypt_suffix[crypt_type]}")
        ret = rknn.export_encrypted_rknn_model(RKNN_MODEL,
                                               RKNN_MODEL_ENCRYPTED,
                                               crypt_level)
        if ret != 0:
            print('Export encrypt rknn model failed!')
            exit(ret)
    print('done')

    rknn.release()

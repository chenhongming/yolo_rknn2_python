import os
import random
import shutil

from pathlib import Path
from tqdm import tqdm


def main():
    nums = 1000

    im_dir = '/home/coco/images/train2017'
    calibration_dir = '/home/database/public/calibration_data/coco'
    os.makedirs(calibration_dir, exist_ok=True)

    samples = os.listdir(im_dir)
    random.shuffle(samples)
    with open(os.path.join(str(Path(calibration_dir).parent), 'calibration.txt'), 'w') as w_file:
        with tqdm(total=nums) as pbar:
            for i in range(nums):
                w_file.writelines(os.path.join(calibration_dir, samples[i]) + '\n')
                shutil.copyfile(os.path.join(im_dir, samples[i]), os.path.join(calibration_dir, samples[i]))
                # display msg
                pbar.set_postfix(**{'filename': samples[i]})
                pbar.update(1)
    print('Done')


if __name__ == '__main__':
    main()

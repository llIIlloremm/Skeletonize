import glob
from tqdm import tqdm

from mmpose.apis import MMPoseInferencer


def main():
    inferencer = MMPoseInferencer('vitpose-s')

    images = glob.glob('images/*.jpg') + glob.glob('images/*.png')

    result_generator = inferencer(images, out_dir='output')
    for result in tqdm(result_generator):
        print(result)



if __name__ == '__main__':
    main()
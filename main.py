from segment import Segmentation
from datasets.dataset import Dataset
from argparse import ArgumentParser
from PIL import Image
import numpy as np
import tifffile as tiff

parser = ArgumentParser()
parser.add_argument("--bs", type=bool, default=False)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--resolution", nargs=2, type=int, default=[224, 224])
parser.add_argument("--activation", type=str, default='selu')
parser.add_argument("--loss_type", type=str, default='DMON')
parser.add_argument("--process", type=str, default='DINO')
parser.add_argument("--threshold", type=float, default=0)
parser.add_argument("--conv_type", type=str, default='ARMA')
parser.add_argument("--image", type=str)
parser.add_argument("--save", type=str)

args = parser.parse_args()

if __name__ == '__main__':
    seg = Segmentation(args.process, args.bs, args.epochs, tuple(args.resolution), args.activation, args.loss_type, args.threshold, args.conv_type)
    img = args.image

    try:
        img = np.asarray(Image.open(img))
    except:
        img = np.array(tiff.imread(img)) 

    mask = np.zeros_like(img)

    _, seg, _ = seg.segment(img, mask)  

    seg *= 255

    binary_image = Image.fromarray(seg)

    binary_image.save(args.save + '.png')


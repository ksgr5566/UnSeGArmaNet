from segment import Segmentation
from datasets.dataset import Dataset
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--bs", type=bool, default=False)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--resolution", nargs=2, type=int, default=[224, 224])
parser.add_argument("--activation", type=str, default='selu')
parser.add_argument("--loss_type", type=str, default='DMON')
parser.add_argument("--process", type=str, default='DINO')
parser.add_argument("--dataset", type=str, default='ECSSD')
parser.add_argument("--threshold", type=float, default=0)
parser.add_argument("--conv_type", type=str, default='ARMA')

args = parser.parse_args()

if __name__ == '__main__':
    seg = Segmentation(args.process, args.bs, args.epochs, tuple(args.resolution), args.activation, args.loss_type, args.threshold, args.conv_type)
    ds = Dataset(args.dataset)

    total_iou = 0
    total_samples = 0
    while ds.loader > 0:
        for img, mask in ds.load_samples():
            try: 
                iou, _, _ = seg.segment(img, mask)
                total_iou += iou
                total_samples += 1
                print(f"IoU for Current Image: {iou:.2f}  mIoU so far: {(total_iou/total_samples):.2f}")

            except Exception as e:
                print(e)
                continue
    
    print(f'Final mIoU: {(total_iou / total_samples):.4f}')


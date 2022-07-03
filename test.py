import os, cv2, argparse
import numpy as np
from pathlib import Path

import torch

import torchvision
from torchvision.transforms import functional as F

import process

parser = argparse.ArgumentParser(
    description='Keypoint Detection Testing Script')
parser.add_argument('--test_path', default='./panda_keypoints/test', type=str,
                    help='path to validation dataset')
parser.add_argument('--weights_path', default='./weights/panda_keypointsrcnn_weights.pth', type=str,
                    help='path to pretrained weights')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use CUDA to train model')
parser.add_argument('--num_classes', default=2, type=int,
                    help='Number of classes per instance')
parser.add_argument('--num_keypoints', default=7, type=int,
                    help='Number of keypoints per instance')
args = parser.parse_args()

def test():
    kpts_ids2names = {0: 'link0', 1: 'link2', 2: 'link3', 3: 'link4', 4: 'link6', 5: 'link7', 6: 'hand'}

    if not os.path.exists('./output'):
        os.mkdir('./output')

    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    model = process.get_model(num_classes = args.num_classes, num_keypoints = args.num_keypoints, weights_path=args.weights_path)

    with torch.no_grad():
        model.to(device)
        model.eval()

        for p in Path(args.test_path).glob('*'): 
            path = str(p)
            name = os.path.basename(path)
            name = '.'.join(name.split('.')[:-1]) + '.png'
            img_original = cv2.imread(path)
            img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB) 
            images = F.to_tensor(img_original)
            images = images.to(device) 
            output = model(images.unsqueeze_(0))

            # print("Predictions: \n", output)
            image = (images[0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
            scores = output[0]['scores'].detach().cpu().numpy()

            high_scores_idxs = np.where(scores > 0.7)[0].tolist() # Indexes of boxes with scores > 0.7
            post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy() # Indexes of boxes left after applying NMS (iou_threshold=0.3)

            # Below, in output[0]['keypoints'][high_scores_idxs][post_nms_idxs] and output[0]['boxes'][high_scores_idxs][post_nms_idxs]
            # Firstly, we choose only those objects, which have score above predefined threshold. This is done with choosing elements with [high_scores_idxs] indexes
            # Secondly, we choose only those objects, which are left after NMS is applied. This is done with choosing elements with [post_nms_idxs] indexes

            keypoints = []
            for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
                keypoints.append([list(map(int, kp[:2])) for kp in kps])

            bboxes = []
            for bbox in output[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
                bboxes.append(list(map(int, bbox.tolist())))
            
            process.visualize(image, bboxes, keypoints, kpts_ids2names, name)

if __name__ == '__main__':
    test()

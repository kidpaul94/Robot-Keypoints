import os, argparse

import torch
from torch.utils.data import DataLoader

import process
from detection.utils import collate_fn
from detection.engine import train_one_epoch, evaluate

parser = argparse.ArgumentParser(
    description='Keypoint Detection Training Script')
parser.add_argument('--training_path', default='./panda_keypoints/train', type=str,
                    help='path to training dataset')
parser.add_argument('--validation_path', default='./panda_keypoints/validation', type=str,
                    help='path to validation dataset')
parser.add_argument('--save_path', default='./weights', type=str,
                    help='Directory for saving checkpoint models')
parser.add_argument('--batch_size', default=8, type=int,
                    help='Batch size for training')
parser.add_argument('--num_epochs', default=10, type=int,
                    help='Number of epochs')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning_rate', default=0.001, type=float,
                    help='Initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum for SGD')
parser.add_argument('--decay', default=0.0005, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.3, type=float,
                    help='For each lr step, what to multiply the lr by')
parser.add_argument('--num_train_backbone', default=None, type=int,
                    help='Number of trainable resnet layers')
parser.add_argument('--num_classes', default=2, type=int,
                    help='Number of classes per instance')
parser.add_argument('--num_keypoints', default=7, type=int,
                    help='Number of keypoints per instance')
args = parser.parse_args()

def custom_train():
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    dataset_train = process.ClassDataset(args.training_path, transform=process.train_transform(), demo=False)
    dataset_valid = process.ClassDataset(args.validation_path, transform=None, demo=False)

    data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    data_loader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = process.get_model(num_classes = args.num_classes, num_keypoints = args.num_keypoints, train_backbone = args.num_train_backbone)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=args.gamma)

    for epoch in range(args.num_epochs):
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=1000)
        lr_scheduler.step()
        evaluate(model, data_loader_valid, device, args.num_keypoints)
        
    # Save model weights after training
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    torch.save(model.state_dict(), os.path.join(args.save_path, 'keypointsrcnn_weights.pth'))

if __name__ == '__main__':
    custom_train()
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.autograd import Variable
from tqdm import tqdm
from PIL import Image

import cityscapes_loader
import metrics

num_classes = 19

mean_std = ([103.939, 116.779, 123.68], [1.0, 1.0, 1.0])
imgTransform = transforms.Compose([transforms.Scale((256,128)),
                                   transforms.ToTensor(),
                                   transforms.Normalize(*mean_std),
                                   transforms.Lambda(lambda x: x.mul_(255))])
target_transform = transforms.Compose([transforms.Scale((256,128),Image.NEAREST)])

valset = cityscapes_loader.CityScapes(mode = 'val', root = 'dataset', transform = imgTransform,
                                      target_transform = target_transform)
valLoader = torch.utils.data.DataLoader(valset, batch_size = 1, shuffle = False, num_workers = 0)

def validate_model(network_path, valLoader, n_classes, use_gpu=False):

    # Setup Model
    model = torch.load(network_path)
    model.eval()

    if use_gpu:
        model = model.cuda()

    gts, preds = [], []
    for i, (images, labels) in tqdm(enumerate(valLoader)):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images)
            labels = Variable(labels)

        outputs = model(images)
        pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy())
        gt = labels.data.cpu().numpy()
        
        for gt_, pred_ in zip(gt, pred):
            gts.append(gt_)
            preds.append(pred_)

    score, class_iou = metrics.scores(gts, preds, n_class=n_classes)

    for k, v in score.items():
        print(k, v)

    for i in range(n_classes):
        print(i, class_iou[i])

validate_model("network.pkl", valLoader, num_classes, use_gpu=True)
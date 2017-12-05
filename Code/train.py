import gc
import torch
import visdom
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from tqdm import tqdm as tqdm

import cityscapes_loader
import loss
import segnet

# Setup visdom for visualization.
vis = visdom.Visdom()
loss_window = vis.line(X=torch.zeros((1)).cpu(),
                       Y=torch.zeros((1)).cpu(),
                       opts=dict(xlabel='Number of Mini Batches',
                                 ylabel='Loss',
                                 title='Training Loss'))

# Load training data.
mean_std = ([103.939, 116.779, 123.68], [1.0, 1.0, 1.0])
imgTransform = transforms.Compose([transforms.Scale((256,128)),
                                   transforms.ToTensor(),
                                   transforms.Normalize(*mean_std),
                                   transforms.Lambda(lambda x: x.mul_(255))])
target_transform = transforms.Compose([transforms.Scale((256,128),Image.NEAREST)])
trainset = cityscapes_loader.CityScapes(mode = 'train', root = 'dataset',transform = imgTransform,
                      target_transform = target_transform)
trainLoader = torch.utils.data.DataLoader(trainset, batch_size = 7, shuffle = True, num_workers = 0)

# Training function.
def train_model(network, criterion, optimizer, trainLoader, n_epochs=10, use_gpu=False):

    if use_gpu:
        # network = torch.nn.DataParallel(network, device_ids=[0, 1])
        network = network.cuda()
        criterion = criterion.cuda()

    # Training loop.
    j=0
    for epoch in range(0, n_epochs):

        gc.collect()

        # Make a pass over the training data.
        t = tqdm(trainLoader, desc='Training epoch %d' % epoch)
        network.train()  # This is important to call before training!
        for (i, (inputs, labels)) in enumerate(t):

            gc.collect()

            # Wrap inputs, and targets into torch.autograd.Variable types.
            inputs = Variable(inputs)
            labels = labels.view(labels.size(0), labels.size(2), labels.size(3))  # To satisfy "CrossEntropyLoss2d"
            labels = Variable(labels)
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # Forward pass:
            outputs = network(inputs)
            loss = criterion(outputs, labels)

            # Backward pass:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            vis.line(
                X=torch.ones((1)).cpu() * j,
                Y=torch.Tensor([loss.data[0]]).squeeze().cpu(),
                win=loss_window,
                update='append')
            j = j + 1

network = segnet.segnet()
criterion = loss.CrossEntropyLoss2d(size_average=True)
optimizer = optim.Adam(network.parameters(), lr=2e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)

# Train the previously defined model.
train_model(network, criterion, optimizer, trainLoader, n_epochs = 10, use_gpu = True)
torch.save(network.cpu(), "network.pkl")
import torch
import numpy as np
import scipy.misc as misc
from PIL import Image
from torch.autograd import Variable

mean_std = ([103.939, 116.779, 123.68], [1.0, 1.0, 1.0])
palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def test_model(network_path, image_path, width, height, output_path, use_gpu=False):

    # Setup image
    img = misc.imread(image_path)

    img = img[:, :, ::-1]
    img = img.astype(np.float64)
    img -= mean_std[0]
    img = misc.imresize(img, (height, width))
    img = img.astype(float) / 255.0
    # NHWC -> NCWH
    img = img.transpose(2, 0, 1) 
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    # Setup Model
    model = torch.load(network_path)
    model.eval()

    if use_gpu:
        model.cuda()
        images = Variable(img.cuda())
    else:
        images = Variable(img)

    outputs = model(images)
    pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy())
    decoded = colorize_mask(pred)
    misc.imsave(output_path, decoded)

test_model("network.pkl", "test.png", 256, 128, "temp.png", use_gpu=True)
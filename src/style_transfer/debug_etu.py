from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms, models


def load_image(img_path, max_size=400, shape=None):
    ''' Load in and transform an image, making sure the image is <= 400 pixels in the x-y dims.'''

    image = Image.open(img_path).convert('RGB')

    # large images will slow down processing
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = in_transform(image)[:3, :, :].unsqueeze(0)

    return image


# helper function for un-normalizing an image and converting it from a Tensor image to a NumPy image for display
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image


def imshow(img):  # Pour afficher une image
    plt.figure(1)
    plt.imshow(img)
    plt.show()


def test():
    vgg = models.vgg19(pretrained=True).features

    # freeze all VGG parameters since we're only optimizing the target image
    for param in vgg.parameters():
        param.requires_grad_(False)

    features = list(vgg)[:23]
    for i, layer in enumerate(features):
        print(i, '\t', layer)


### Run an image forward through a model and get the features for a set of layers. 'model' is supposed to be vgg19
def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv0',
                  '5': 'conv5',
                  '10': 'conv10',
                  '19': 'conv19',  ## content representation
                  }

    features = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features


def gram_matrix(tensor):
    # tensor: Nfeatures x H x W ==> M = Nfeatures x Npixels with Npixel=HxW
    _, n_features, h, w = tensor.size()
    tensor = tensor.view(n_features, h * w)
    return torch.mm(tensor, torch.transpose(tensor, 0, 1))


def stylize(content, style, content_coeff=0.5, style_coeff=0.5):
    total_coeff = content_coeff + style_coeff
    content_coeff = content_coeff / total_coeff
    style_coeff = style_coeff / total_coeff

    target = content.clone().requires_grad_(True).to(device)
    model = models.vgg19(pretrained=True).features
    model.to(device)


    optimizer = optim.Adam([target], lr=0.003)
    for i in range(5):
        # get the features from your target image
        features = get_features(target, model)
        content_features = get_features(content, model, {'19': 'conv19'})
        style_features = get_features(style, model, {'0': 'conv0',
                                                 '5': 'conv5',
                                                 '10': 'conv10'})

        # the content loss
        content_loss = 0
        for key, F in content_features.items():
            P = features[key]
            content_loss += torch.mean((F - P) ** 2)

        # the style loss
        style_loss = 0
        for key, F in style_features.items():
            P = features[key]
            G = gram_matrix(F)
            A = gram_matrix(P)
            style_loss += torch.mean((G - A) ** 2)

        # calculate the *total* loss
        total_loss = content_coeff*content_loss + style_coeff*style_loss

        # update your target image
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        print("Content loss : ", content_loss)
        print("Style loss : ", style_loss)
        print("Total loss: ", total_loss)

    return im_convert(target)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(device)

    ########################## DISPLAY IMAGE#########################################################""
    content = load_image('src/style_transfer/images/montagne_small.jpg').to(device)
    style = load_image('src/style_transfer/images/peinture1_small.jpg', shape=content.shape[-2:]).to(device)

    target = stylize(content, style)
    print(type(target))
    imshow(target)

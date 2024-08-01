from torchvision import models, transforms
import torch
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

def predict(image_path):
    model = timm.create_model(
        "vit_base_patch16_224",
        num_classes=2,
        checkpoint_path="./model_best.pth.tar"
    )

    #https://pytorch.org/docs/stable/torchvision/models.html
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))


    img = Image.open(image_path)
    batch_t = torch.unsqueeze(transform(img), 0)

    model.eval()
    out = model(batch_t)
    #print(out)

    with open('imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    value, index = torch.max(prob, 0)
    return (classes[index.item()], value.item())




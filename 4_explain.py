import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# = Build model =
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("casting_defect_model.pth", map_location=device, weights_only=True))
model = model.to(device)
model.eval()


# = Preperation =
# Image
image_path = "casting_data/test/def_front/cast_def_0_40.jpeg"
raw_image = Image.open(image_path).convert("RGB").resize((224,224))

# Convert image into number between 0.0 and 1.0
rgb_img = np.float32(raw_image) / 255.0

# Convert into tensor
test_transforms = transforms.Compose([
    transforms.ToTensor()
])
input_tensor = test_transforms(rgb_img).unsqueeze(0).to(device)


# = Generate Heatmap =
# = Attach probe =
target_layer = [model.layer4[-1]] # layer4 is the final layer of ResNet18
cam = GradCAM(model=model, target_layers=target_layer)

targets = [ClassifierOutputTarget(0)] # 0 is the first class "Defective"; 1 is the second class "OK"
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0,:] # Run calculus backwards to get the heatmap
visualize = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True) # Paint heatmap on the image.


# = Output =
fig, axes = plt.subplots(1, 2, figsize=(10,5))

axes[0].imshow(raw_image)
axes[0].set_title("Normal Image")
axes[0].axis("off")

axes[1].imshow(visualize)
axes[1].set_title("W/ Heatmap")
axes[1].axis("off")

plt.show() 


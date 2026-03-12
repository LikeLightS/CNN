import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pylab as plt

# = Rebuild model =
model = models.resnet18()
num_feature = model.fc.in_features
model.fc = nn.Linear(num_feature, 2)

# Load trained model
save_path = "casting_defect_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
model = model.to(device)

# Turn to evaluation mode (Testing not training)
model.eval()


# = Load Data =
data_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])
image_path = "casting_data/test/def_front/cast_def_0_224.jpeg"
raw_image = Image.open(image_path).convert('RGB') # Open using PIL
image_tensor = data_transforms(raw_image) # Transform the image

image_tensor = image_tensor.unsqueeze(0).to(device) # Add the batch size on the front, which is 1 currently. [1, 3, 224, 224]

# No need derivitive, so close it
with torch.no_grad():
    output = model(image_tensor) # Output 2 number
    prediction_index = torch.argmax(output, dim=1).item() # Choose the highest number's index, 0 or 1

class_name = ["def_front", "ok_front"]
prediction_label = class_name[prediction_index]

plt.imshow(raw_image)
plt.title(f"Prediction: {prediction_label}")
plt.axis("off")
plt.show()
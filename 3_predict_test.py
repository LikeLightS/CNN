import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
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


# = Load Data =
data_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])
test_dataset = datasets.ImageFolder(root="casting_data/test", transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
class_name = test_dataset.classes


# = Test Data =
# Turn to evaluation mode (Testing not training)
model.eval()

total = 0
correct = 0
with torch.no_grad(): # No need derivitive, so close it
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        output = model(images) # Output 2 number
        prediction_index = torch.argmax (output, dim=1) # Choose the highest number's index, 0 or 1

        total += labels.size(0)
        correct += (prediction_index == labels).sum().item()

accuracy = correct/total * 100
print(f"Accuracy: {accuracy}%. Prediction: {correct} out of {total} is correct.")
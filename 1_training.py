import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# = Data =
# Resize image from 512 to 224
data_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])
# Load data
train_dataset = datasets.ImageFolder(root="casting_data/train", transform=data_transforms)
train_loader = DataLoader (train_dataset, batch_size = 32, shuffle=True)

# = Verify =
import matplotlib.pyplot as plt
# Verify the data being loaded, make sure the image is loaded correctly
# images, labels = next(iter(train_loader))

# plt.imshow(images[0].permute(1,2,0))
# plt.title(train_dataset.classes[labels[0]])
# plt.show()


from torchvision import models
import torch.nn as nn

# = Model =
# Load pretrained ResNet18 model - This pretrained model knows shape, edge, line, etc
model = models.resnet18(weights="DEFAULT")

# Freeze base layer (prevent touching what the model already knows)
for param in model.parameters():
    param.requires_grad = False


# = Feature =
# Change model's head to fit 2 classes (Def and OK)
num_features = model.fc.in_features
# Change the "num_features" of classes into 2 classes
model.fc = nn.Linear(num_features, 2) # model.fc is the Fully Connected Layer, aka the last layer


# = Optimization =
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001) # Optimize only the last layer


# = Training =
# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Start the training
model.train()
epochs = 3
print("Starting Training...")

for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:

        # Move image/label to GPU
        images = images.to(device)
        labels = labels.to(device)

        # Zero gradient: Wipe math board from last batch
        optimizer.zero_grad()

        # Forward pass: Give images to model to predict
        outputs = model(images)

        # Calculate loss from prediction
        loss = criterion(outputs, labels)

        # Backward pass: Calculate the derivatives
        loss.backward()

        # Optimize: improve model
        optimizer.step()
        
        running_loss += loss.item() # See whether it is improving or not
    
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}")

print("Training Complete !!!")

# Save the trained model
save_path = "casting_defect_model.pth"
torch.save(model.state_dict(), save_path) # state_dict() is a dict that contain the bias and weights of the model
print(f"Model saved to {save_path}")

# pth is older format while safetensors is newer
# pth can be dangerous as it uses Pickle (python feature) and stores in Python dict format
# Running pth is running code to rebuild the file, code can be malicious
# safetensors stores in numbers only, not code.

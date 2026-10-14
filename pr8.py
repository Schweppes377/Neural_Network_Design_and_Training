import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image, ImageOps
import os
import matplotlib.pyplot as plt


MY_TRANSFORM = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # первый слой - простые признаки
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        # второй слой - сложные признаки
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # пулинг для концентрации важных признаков
        self.pool = nn.MaxPool2d(2, 2)
        # классификация
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #тензор в вектор
        x = x.view(-1, 32 * 7 * 7)
        # классификация
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, filename):
    print("Обучаем модель с нуля...")
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=MY_TRANSFORM)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    # 3 эпохи обучения для повышения точности
    for epoch in range(3):
        total_correct = 0
        total_items = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(output.data, 1)
            total_items += target.size(0)
            total_correct += (predicted == target).sum().item()

        accuracy = 100 * total_correct / total_items
        print(f"Эпоха {epoch + 1} пройдена. Точность этапа: {accuracy:.2f}%")

    print("Веса сохранены.")
    torch.save(model.state_dict(), filename)

def predict_user_image(model, image_path):
    if not os.path.exists(image_path):
        print(f"Файл {image_path} не найден!")
        return

    img = Image.open(image_path).convert('L')
    img = ImageOps.invert(img)
    img_tensor = MY_TRANSFORM(img).unsqueeze(0)

    plt.figure(figsize=(2, 2))
    plt.imshow(img_tensor.squeeze().numpy(), cmap='gray')
    plt.title("Что видит нейросеть")
    plt.axis('off')
    plt.show()

    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)

        probs, indices = torch.topk(probabilities, 3)

        for i in range(3):
            digit = indices[0][i].item()
            prob = probs[0][i].item() * 100
            print(f"Это цифра {digit} (уверенность {prob:.2f}%)")


if __name__ == "__main__":
    net = SimpleCNN()
    file_weights = 'mnist_cnn.pth'
    file_image = 'digit7.png'

    if not os.path.exists(file_weights):
        train_model(net, file_weights)
    else:
        print(f"Веса уже подготовлены. Загрузка {file_weights}...")
        net.load_state_dict(torch.load(file_weights))

    predict_user_image(net, file_image)
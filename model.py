from torch import nn
import torchvision


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.cnn1 = torchvision.models.resnet50(pretrained=False)
        self.fc_in_features = self.cnn1.fc.in_features

        self.cnn1.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(1, 1), padding=(1, 1))
        self.cnn1 = nn.Sequential(*(list(self.cnn1.children())[:-1]))

        #
        # self.cnn1 = nn.Sequential(
        #     nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #
        #     nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        # )

        self.fc1 = nn.Sequential(
            nn.Linear(self.fc_in_features * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward_once(self, x):
        # Forward pass
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.forward_once(input1)
        # forward pass of input 2
        output2 = self.forward_once(input2)
        return output1, output2

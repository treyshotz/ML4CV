from torch import nn
import torchvision


class NonSiameseNetwork(nn.Module):
    def __init__(self):
        super(NonSiameseNetwork, self).__init__()

        #By design of the dataset input 1 will always be mnist and input 2 will always be svhn

        self.cnn1 = torchvision.models.resnet50(pretrained=False)
        self.fc_in_features = self.cnn1.fc.in_features
        self.cnn1.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(1, 1), padding=(1, 1))
        self.cnn1 = nn.Sequential(*(list(self.cnn1.children())[:-1]))

        self.cnn2 = torchvision.models.resnet50(pretrained=False)
        self.cnn2.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(1, 1), padding=(1, 1))
        self.cnn2 = nn.Sequential(*(list(self.cnn2.children())[:-1]))

        self.fc1 = nn.Sequential(
            nn.Linear(self.fc_in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.fc_in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10),
        )

    def forward_once(self, x):
        # Forward pass
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward_second(self, x):
        output = self.cnn2(x)
        output = output.view(output.size()[0], -1)
        output = self.fc2(output)
        return output


    def forward(self, input1, input2):
        # forward pass of mnist input
        output1 = self.forward_once(input1)
        # forward pass of svhn input
        output2 = self.forward_second(input2)
        return output1, output2

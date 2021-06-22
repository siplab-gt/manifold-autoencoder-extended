import torch
import torch.nn as nn
import torch.nn.functional as F

class SimplifiedResNet(nn.Module):
    def __init__(self, original_model, image_dim):
        super(SimplifiedResNet, self).__init__()
        if image_dim == 64:
            self.features = nn.Sequential(*list(original_model.children())[:-1])
        else:
            self.features = nn.Sequential(*list(original_model.children())[:-2])

    def forward(self, x):
        x = self.features(x)
        return x

class CNN(nn.Module):

    def __init__(self, y_dim):
        """
        Initialize classifier
        Inputs:
        - y_dim : number of classes
        """
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, y_dim)

    def forward(self, x):
        """
        Perform classification using the CNN classifier

        Inputs:
        - x : input data sample

        Outputs:
        - out: unnormalized output
        - prob_out: probability output
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        out = self.fc2(x)
        prob_out = F.softmax(out, dim=1)
        return prob_out, out

class LeNet(nn.Module):

    def __init__(self,y_dim,img_sz = 28):
        """
        Initialize classifier
        Inputs:
        - y_dim : number of classes
        """
        feat_map_sz = img_sz//4
        self.n_feat = 50*feat_map_sz * feat_map_sz
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, padding=2)
        self.conv2 = nn.Conv2d(20, 50, 5,  padding=2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(self.n_feat, 128)
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """
        Perform classification using the CNN classifier

        Inputs:
        - x : input data sample

        Outputs:
        - output: unnormalized output
        - prob_out: probability output
        """
        x = self.conv1(x)
        x = F.max_pool2d(F.relu(x), (2, 2))
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.reshape(x,(-1, self.n_feat))
        x = self.fc1(x)
        fc_out1_128 = F.relu(x)
        x = self.fc2(fc_out1_128)
        fc_out2_84 = F.relu(x)
        output = self.fc3(fc_out2_84)
        prob_out = F.softmax(output)

        return prob_out,output

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

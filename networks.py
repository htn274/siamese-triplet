import torch.nn as nn
import torch.nn.functional as F


class EmbeddingNet(nn.Module):
    def __init__(self, emb_dims):
        super(EmbeddingNet, self).__init__()
        self.emb_dims = emb_dims
        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 3, padding='same'), nn.ReLU(), #[29, 29, 1] -> [29, 29, 32]
                                     nn.MaxPool2d(2, stride=2),                      #[29, 29, 32] -> [14, 14, 32]
                                     nn.Conv2d(32, 32, 3, padding='same'), nn.ReLU(), #[14, 14, 32] -> [14, 14, 32]
                                     nn.MaxPool2d(2, stride=2),                       #[14, 14, 32] -> [7, 7, 32]
                                     nn.Conv2d(32, 64, 3, padding='same'), nn.ReLU(), #[7, 7, 32] -> [7, 7, 64]
                                     nn.MaxPool2d(2, stride=2),                       #[3, 3, 64]
                                     )

        self.fc = nn.Sequential(nn.Linear(64 * 3 * 3, 256),
                                nn.ReLU(),
                                nn.Linear(256, 256),
                                nn.ReLU(),
                                nn.Linear(256, emb_dims))
        
#         self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
#                                      nn.MaxPool2d(2, stride=2),
#                                      nn.Conv2d(32, 64, 5), nn.PReLU(),
#                                      nn.MaxPool2d(2, stride=2))

#         self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
#                                 nn.PReLU(),
#                                 nn.Linear(256, 256),
#                                 nn.PReLU(),
#                                 nn.Linear(256, 2)
#                                 )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        #print(output.shape)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2, self).__init__()

    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)


class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(embedding_net.emb_dims, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))
    
    def get_prediction(self, x):
        return self.fc1(self.get_embedding(x))

class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)

class MyModel(nn.Module):
    def __init__(self, pretrained_model):
        super(MyModel, self).__init__()
        self.pretrained_model = pretrained_model
        self.last_layer = nn.Sigmoid()

    def forward(self, x):
        return self.last_layer(self.pretrained_model(x))

modelll = models.resnet18(pretrained = False)
modelll.fc = nn.Linear(fc_features,384)
model = MyModel(modelll)

import torch
import torch.nn.functional as F


class FeatureExtractor:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """
    def __init__(self, model, feature_extractor, classifier, target_layers):
        self.model = model
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        # for name, module in self.model._modules.items():
        for name, module in self.feature_extractor._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs:
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """
    def __init__(self, model, feature_extractor, classifier, target_layers):
        self.model = model
        # self.feature_extractor = FeatureExtractor(self.model.conv5, target_layers)
        self.feature_extractor = FeatureExtractor(self.model, feature_extractor, classifier, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output  = self.feature_extractor(x)
        conv5_pool = self.model.pool(output).squeeze()
        logits = self.model.classifier(conv5_pool)
        output = logits
        return target_activations, output


class GradCam:
    def __init__(self, model, feature_extractor, classifier, target_layer_names):
        self.model = model
        self.flag = model.training
        self.model.eval()
        # self.cuda = use_cuda
        # if self.cuda:
        #     self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, feature_extractor, classifier, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index = None):
        input.requires_grad=True
        # if self.cuda:
        #     features, output = self.extractor(input.cuda())
        # else:
        features, output = self.extractor(input)

        if output.dim()==1:
            output = output.unsqueeze(0)

        if index is None:
            index = torch.argmax(output, dim=-1)

        one_hot = torch.zeros_like(output)
        one_hot[[torch.arange(output.size(0)), index]] = 1
        one_hot = torch.sum(one_hot.to(input.device) * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        grads_val = self.extractor.get_gradients()[-1]
        grads_val = F.relu(grads_val)
        weights = grads_val.mean(-1).mean(-1)

        if self.flag:
            self.model.train()
        return weights.clone().detach()


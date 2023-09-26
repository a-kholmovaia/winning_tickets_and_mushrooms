import torch 
import torch.nn as nn
import copy 
class MushroomClassifier(nn.Module):
    HIDDEN1_SIZE = 128
    HIDDEN2_SIZE = 256
    HIDDEN3_SIZE = 64
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(21, self.HIDDEN1_SIZE)
        # initializing layers with he initialization
        nn.init.kaiming_uniform_(self.hidden1.weight, mode='fan_in', nonlinearity='relu')
        #torch.nn.init.normal_(self.hidden1.weight, mean=0.0, std=0.01)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(self.HIDDEN1_SIZE, self.HIDDEN2_SIZE)
        nn.init.kaiming_uniform_(self.hidden2.weight, mode='fan_in', nonlinearity='relu')
        #torch.nn.init.normal_(self.hidden2.weight, mean=0.0, std=0.01)
        self.act2 = nn.ReLU()
        self.hidden3 = nn.Linear(self.HIDDEN2_SIZE, self.HIDDEN3_SIZE)
        nn.init.kaiming_uniform_(self.hidden3.weight, mode='fan_in', nonlinearity='relu')
        #torch.nn.init.normal_(self.hidden3.weight, mean=0.0, std=0.01)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(self.HIDDEN3_SIZE, 1)
        nn.init.kaiming_uniform_(self.output.weight, mode='fan_in', nonlinearity='relu')
        #torch.nn.init.normal_(self.output.weight, mean=0.0, std=0.01)
        self.act_output = nn.Sigmoid()

        self.hidden1.mask = nn.Parameter(torch.ones(self.HIDDEN1_SIZE, 21), 
                                         requires_grad=False)
        self.hidden2.mask = nn.Parameter(torch.ones(self.HIDDEN2_SIZE, self.HIDDEN1_SIZE), 
                                         requires_grad=False)
        self.hidden3.mask = nn.Parameter(torch.ones(self.HIDDEN3_SIZE, self.HIDDEN2_SIZE), 
                                         requires_grad=False)
        self.output.mask = nn.Parameter(torch.ones(1, self.HIDDEN3_SIZE), 
                                         requires_grad=False)
        self.init_dict = copy.deepcopy(self.state_dict())
        

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act3(self.hidden3(x))
        x = self.act_output(self.output(x))
        return x

    def prune_weights(self, layer_name):
      for name, module in self.named_modules():
          if layer_name == name:
            with torch.no_grad():
              module.weight.data[module.mask == 0] = 0
              module.weight.data[module.mask == 0].requires_grad_(False)

    def rewind_weights(self):
       for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
               init_weights_layer = torch.tensor(self.init_dict[name + '.weight']).cuda()
               module.weight.data[module.mask != 0] = init_weights_layer[module.mask != 0]

    def get_module(self, name1):
      for name, module in self.named_modules():
          if name1 == name:
            return module
    
    def get_sparcity(self):
        not_pruned_weights = 0
        total_weights = 0
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                not_pruned_weights += torch.count_nonzero(module.mask)
                total_weights += torch.numel(module.mask)
        self.sparcity = 1-not_pruned_weights/total_weights
        return self.sparcity
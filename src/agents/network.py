import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, input_dim, action_size, seed=42):
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # input_dim sarà la somma di tutte le feature (es. 49 per ora)
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_size)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)



    def get_description(self):
        """Restituisce una stringa formattata con i dettagli dei layer."""
        info = []
        for name, layer in self.named_children():
            if isinstance(layer, nn.Linear):
                info.append(f"Layer ({name}): Input {layer.in_features} -> Neuroni {layer.out_features}")
        return "\n".join(info)
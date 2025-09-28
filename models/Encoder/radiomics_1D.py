import torch
import torch.nn as nn

class Radiomics1DNet(nn.Module):
    
    def __init__(self, radio_input_dim: int, n_classes: int = 2, hidden_dims: list = [ 256, 128, 64], 
                 dropout_rate: float = 0.2, batch_norm: bool = True):
        super(Radiomics1DNet, self).__init__()
        
        self.num_classes = n_classes
        layers = []
        
        # Input layer
        current_dim = radio_input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(*layers)
        
        # Task-specific output layer
        self.classifier = nn.Linear(current_dim, n_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.01)  # Smaller gain
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        
    def forward(self, x_mri, x_path, x_omic):
        """Forward pass"""
        features = self.feature_extractor(x_mri)
        logits = self.classifier(features)
        return logits
    
    # def get_features(self, x):
    #     """Extract learned features (useful for analysis)"""
    #     return self.feature_extractor(x)

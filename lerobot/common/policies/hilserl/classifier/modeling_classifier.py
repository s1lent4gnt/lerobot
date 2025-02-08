import logging
from typing import Optional, Dict, List

import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor, nn

from .configuration_classifier import ClassifierConfig


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def safe_key(key: str) -> str:
    return key.replace(".", "_")


class ClassifierOutput:
    """Wrapper for classifier outputs with additional metadata."""

    def __init__(
        self, logits: Tensor, probabilities: Optional[Tensor] = None, hidden_states: Optional[Tensor] = None
    ):
        self.logits = logits
        self.probabilities = probabilities
        self.hidden_states = hidden_states

    def __repr__(self):
        return (
            f"ClassifierOutput(logits={self.logits}, "
            f"probabilities={self.probabilities}, "
            f"hidden_states={self.hidden_states})"
        )


class SpatialLearnedEmbeddings(nn.Module):
    def __init__(self, height, width, channel, num_features=8):
        """
        PyTorch implementation of learned spatial embeddings
        
        Args:
            height: Spatial height of input features
            width: Spatial width of input features
            channel: Number of input channels
            num_features: Number of output embedding dimensions
        """
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel
        self.num_features = num_features

        self.kernel = nn.Parameter(torch.empty(channel, height, width, num_features))
        
        nn.init.kaiming_normal_(self.kernel, mode='fan_in', nonlinearity='linear')

    def forward(self, features):
        """
        Forward pass for spatial embedding
        
        Args:
            features: Input tensor of shape [B, H, W, C] or [H, W, C] if no batch
        Returns:
            Output tensor of shape [B, C*F] or [C*F] if no batch
        """

        features = features.last_hidden_state

        original_shape = features.shape
        if features.dim() == 3:
            features = features.unsqueeze(0)  # Add batch dim

        features_expanded = features.unsqueeze(-1)  # [B, H, W, C, 1]
        kernel_expanded = self.kernel.unsqueeze(0)  # [1, H, W, C, F]

        # Element-wise multiplication and spatial reduction
        output = (features_expanded * kernel_expanded).sum(dim=(2, 3))  # Sum H,W
        
        # Reshape to combine channel and feature dimensions
        output = output.view(output.size(0), -1)  # [B, C*F]

        # Remove batch dim
        if len(original_shape) == 3:
            output = output.squeeze(0)

        return output


class Classifier(
    nn.Module,
    PyTorchModelHubMixin,
    # Add Hub metadata
    library_name="lerobot",
    repo_url="https://github.com/huggingface/lerobot",
    tags=["robotics", "vision-classifier"],
):
    """Image classifier built on top of a pre-trained encoder."""

    # Add name attribute for factory
    name = "classifier"

    def __init__(self, config: ClassifierConfig):
        super().__init__()
        self.config = config

        self.encoders = nn.ModuleDict()
        self.processors = {}  # For non nn.Module objects

        for image_key in config.image_keys:
            encoder, processor = self._create_single_encoder(image_key)
            self.encoders[safe_key(image_key)] = encoder
            self.processors[safe_key(image_key)] = processor

        self._build_classifier_head()

    def _create_single_encoder(self, image_key: str):
        from transformers import AutoImageProcessor, AutoModel

        # Initialize processor and base model
        processor = AutoImageProcessor.from_pretrained(
            self.config.model_name, 
            trust_remote_code=True
        )
        encoder = AutoModel.from_pretrained(
            self.config.model_name, 
            trust_remote_code=True
        )

        # Handle multimodal models
        if hasattr(encoder, "vision_model"):
            encoder = encoder.vision_model

        # Freeze encoder parameters
        for param in encoder.parameters():
            param.requires_grad = False

        if self.config.model_type == "cnn":
            if hasattr(encoder, "fc"):
                feature_dim = encoder.fc.in_features
                encoder = nn.Sequential(*list(encoder.children())[:-1])
            else:
                feature_dim = encoder.config.hidden_sizes[-1]

            encoder = nn.Sequential(
                encoder,
                SpatialLearnedEmbeddings(
                    height=4, 
                    width=4, 
                    channel=feature_dim,
                    num_features=8
                ),
                nn.Dropout(self.config.dropout_rate),
                nn.Linear(feature_dim * 8, 256),
                nn.LayerNorm(256),
                nn.Tanh()
            )
            
        return encoder, processor

    def _build_classifier_head(self) -> None:
        """Initialize the classifier head architecture."""
        # Get input dimension based on model type
        # if self.is_cnn:
        #     input_dim = self.feature_dim
        # else:  # Transformer models
        #     if hasattr(self.encoder.config, "hidden_size"):
        #         input_dim = self.encoder.config.hidden_size
        #     else:
        #         raise ValueError("Unsupported transformer architecture since hidden_size is not found")

        input_dim = 256 * len(self.config.image_keys)  # 256 per encoder

        self.classifier_head = nn.Sequential(
            # Binary Classifier
            nn.Linear(input_dim, 256),
            nn.Dropout(self.config.dropout_rate),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1 if self.config.num_classes == 2 else self.config.num_classes),
        )
        self.classifier_head = self.classifier_head.to(self.config.device)


    def _get_encoder_output(self, x: torch.Tensor, image_key: str) -> torch.Tensor:
        """Process input through a specific encoder."""
        key = safe_key(image_key)
        processor = self.processors[key]
        encoder = self.encoders[key]
        
        processed = processor(
            images=x,
            return_tensors="pt"
        )["pixel_values"].to(x.device)
        
        return encoder(processed)

    def forward(self, inputs: Dict[str, Tensor]) -> ClassifierOutput:
        """Forward pass through all encoders."""
        # Process each image source through its encoder
        encoder_outputs = []
        for image_key in self.config.image_keys:
            if image_key in inputs:
                features = self._get_encoder_output(inputs[image_key], image_key)
                encoder_outputs.append(features)
                
        combined = torch.cat(encoder_outputs, dim=-1)
        
        logits = self.classifier_head(combined)

        logits = logits.squeeze(1)
        
        return ClassifierOutput(
            logits=logits,
            probabilities=torch.sigmoid(logits) if self.config.num_classes == 2 
                          else torch.softmax(logits, dim=-1),
            hidden_states=combined
        )

    def predict_reward(self, x):
        if self.config.num_classes == 2:
            return (self.forward(x).probabilities > 0.5).float()
        else:
            return torch.argmax(self.forward(x).probabilities, dim=1)

import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests
import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["HTTP_PROXY"] = "127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "127.0.0.1:7890"

def get_loss(params, curr_data, variables, iter_time_idx, loss_weights, use_sil_for_loss,
             sil_thres, use_l1, ignore_outlier_depth_loss, tracking=False, 
             mapping=False, do_ba=False, plot_dir=None, visualize_tracking_loss=False, tracking_iteration=None):
    # Initialize Loss Dictionary
    losses = {}

    # Existing loss calculation code...
    
    # Feature Extraction with DINOv2
    dino_model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
        features_observed = dino_model(curr_data['im'].unsqueeze(0))  # Add batch dimension if necessary
        features_rendered = dino_model(im.unsqueeze(0)) #TODO

    # Calculate Feature Metric Loss (using L2 norm for simplicity)
    feature_loss = torch.norm(features_observed - features_rendered, p=2)
    
    # Include feature metric loss in the losses dictionary
    losses['feature_metric'] = feature_loss
    
    # Existing visualization and loss weighting code...

    return loss, variables, weighted_losses


url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
dino_model = AutoModel.from_pretrained('facebook/dinov2-base')

inputs = processor(images=image, return_tensors="pt")
outputs = dino_model(**inputs)
last_hidden_states = outputs[0]

# We have to force return_dict=False for tracing
dino_model.config.return_dict = False

with torch.no_grad():
    traced_model = torch.jit.trace(dino_model, [inputs.pixel_values])
    traced_outputs = traced_model(inputs.pixel_values)

print((last_hidden_states - traced_outputs[0]).abs().max())
#import libraries
from transformers import AutoProcessor, SamModel
processor = AutoProcessor.from_pretrained("facebook/sam-vit-huge")
model = SamModel.from_pretrained("facebook/sam-vit-huge")
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.transforms.functional import resize, to_pil_image
from typing import Tuple
import numpy as np
import cv2
import torch
from skimage.feature import peak_local_max


#helper functions
def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
    """
    Compute the output size given input size and target long side length.
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)
def preprocess(x: torch.Tensor, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375],
               img_size=1024) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""

    pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
    pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)

    # Normalize colors
    x = (x - pixel_mean) / pixel_std

    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x
def prepare_mask(image, target_length=1024):
    target_size = get_preprocess_shape(image.shape[0], image.shape[1], target_length)
    mask = np.array(resize(to_pil_image(image), target_size))

    input_mask = torch.as_tensor(mask)
    input_mask = input_mask.permute(2, 0, 1).contiguous()[None, :, :, :]

    input_mask = preprocess(input_mask)

    return input_mask
def point_selection(mask_sim, topk=5):
    # Top-1 point selection
    w, h = mask_sim.shape
    topk_xy = mask_sim.flatten(0).topk(topk)[1]
    topk_x = (topk_xy // h).unsqueeze(0)
    topk_y = (topk_xy - topk_x * h)
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
    topk_label = np.array([1] * topk)
    topk_xy = topk_xy.cpu().numpy()

    # Top-last point selection
    last_xy = mask_sim.flatten(0).topk(topk, largest=False)[1]
    last_x = (last_xy // h).unsqueeze(0)
    last_y = (last_xy - last_x * h)
    last_xy = torch.cat((last_y, last_x), dim=0).permute(1, 0)
    last_label = np.array([0] * topk)
    last_xy = last_xy.cpu().numpy()

    return topk_xy, topk_label, last_xy, last_label
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255, 255, 0, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


#Load Image,reference image and test image
ref_image = cv2.imread("golden.png", cv2.IMREAD_COLOR)
test_image=cv2.imread("3.png", cv2.IMREAD_COLOR)
ref_mask= cv2.imread("sam_mask/sam_mask_2.png", cv2.IMREAD_COLOR)

#convert into tensors
pixel_values = processor(images=ref_image, return_tensors="pt").pixel_values

#move model to cuda
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Step 1: Image features encoding
with torch.no_grad():
  ref_feat = model.get_image_embeddings(pixel_values.to(device))
  ref_feat = ref_feat.squeeze().permute(1, 2, 0)

# Step 2: interpolate reference mask
ref_mask = prepare_mask(ref_mask)
ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0: 2], mode="bilinear")
ref_mask = ref_mask.squeeze()[0]

# Step 3: Target feature extraction
target_feat = ref_feat[ref_mask > 0]
target_embedding = target_feat.mean(0).unsqueeze(0)
target_feat = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
target_embedding = target_embedding.unsqueeze(0)

# prepare test image for the model
inputs = processor(images=test_image, return_tensors="pt").to(device)
pixel_values = inputs.pixel_values

# image feature encoding
with torch.no_grad():
  test_feat = model.get_image_embeddings(pixel_values).squeeze()

# Cosine similarity
num_channels, height, width = test_feat.shape
test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
test_feat_reshaped = test_feat.reshape(num_channels, height * width)
sim = target_feat @ test_feat_reshaped
sim = sim.reshape(1, 1, height, width)
sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
sim = processor.post_process_masks(sim.unsqueeze(1), original_sizes=inputs["original_sizes"].tolist(), reshaped_input_sizes=inputs["reshaped_input_sizes"].tolist(),
                                   binarize=False)
sim = sim[0].squeeze()

#Positive-negative location prior
topk_xy_i, topk_label_i, last_xy_i, last_label_i = point_selection(sim, topk=5)
topk_xy = np.concatenate([topk_xy_i, last_xy_i], axis=0)
topk_label = np.concatenate([topk_label_i, last_label_i], axis=0)

# Obtain the target guidance for cross-attention layers
sim = (sim - sim.mean()) / torch.std(sim)
similarity_map = sim
sim = F.interpolate(sim.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear")
attention_similarity = sim.sigmoid_().unsqueeze(0).flatten(3)

# Normalize similarity map
sim_map_np = similarity_map
if isinstance(similarity_map, torch.Tensor):
    sim_map_np = similarity_map.cpu().numpy()
sim_map_np = (sim_map_np - sim_map_np.min()) / (sim_map_np.max() - sim_map_np.min())

# Detect peak points (local maxima)
coordinates = peak_local_max(
    sim_map_np,
    min_distance=10,         # distance between detected peaks
    threshold_abs=0.5,       # only pick strong peaks
    num_peaks=10            # limit max number of instances
)

# Visualize the detected points on similarity map
plt.figure(figsize=(10, 6))
plt.imshow(sim_map_np, cmap='jet')
for y, x in coordinates:
    plt.plot(x, y, 'ro')  # Red dot at each peak
plt.title("Detected Points from Similarity Map")
plt.axis("off")
plt.show()

input_points = [[ [int(x), int(y)] for y, x in coordinates ]]  # Note: order is [x, y]
input_labels = [[1] * len(coordinates)]  # All are foreground

# prepare test image and prompts for the model
inputs = processor(test_image, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(device)

# First-step prediction
with torch.no_grad():
  outputs = model(
      input_points=inputs.input_points,
      input_labels=inputs.input_labels,
      image_embeddings=test_feat.unsqueeze(0),
      multimask_output=False,
      attention_similarity=attention_similarity,  # Target-guided Attention
      target_embedding=target_embedding  # Target-semantic Prompting
  )
  best_idx = 0

# Cascaded Post-refinement-1
with torch.no_grad():
  outputs_1 = model(
              input_points=inputs.input_points,
              input_labels=inputs.input_labels,
              input_masks=outputs.pred_masks.squeeze(1)[best_idx: best_idx + 1, :, :],
              image_embeddings=test_feat.unsqueeze(0),
              multimask_output=True)

#Cascaded Post-refinement-2
masks = processor.image_processor.post_process_masks(outputs_1.pred_masks.cpu(),
                                                     inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0].squeeze().numpy()

best_idx = torch.argmax(outputs_1.iou_scores).item()
y, x = np.nonzero(masks[best_idx])
x_min = x.min()
x_max = x.max()
y_min = y.min()
y_max = y.max()
input_boxes = [[[x_min, y_min, x_max, y_max]]]

inputs = processor(test_image, input_points=[topk_xy.tolist()], input_labels=[topk_label.tolist()], input_boxes=input_boxes,
                   return_tensors="pt").to(device)

final_outputs = model(
    input_points=inputs.input_points,
    input_labels=inputs.input_labels,
    input_boxes=inputs.input_boxes,
    input_masks=outputs_1.pred_masks.squeeze(1)[:,best_idx: best_idx + 1, :, :],
    image_embeddings=test_feat.unsqueeze(0),
    multimask_output=True)

masks = processor.image_processor.post_process_masks(final_outputs.pred_masks.cpu(),
                                                     inputs["original_sizes"].cpu(),
                                                     inputs["reshaped_input_sizes"].cpu())[0].squeeze().numpy()

fig, axes = plt.subplots()
best_idx = torch.argmax(final_outputs.iou_scores).item()
axes.imshow(np.array(test_image))
show_mask(masks[best_idx], axes,random_color=False)
axes.title.set_text(f"Predicted mask")
axes.axis("off")
plt.show()

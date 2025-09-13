import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def plot_image(image):
    plt.imshow(image)
    plt.axis('off')  # Hide axis
    plt.show()


def show_masks(image, masks):
    """Visualizes each mask individually with its label as the title."""
    for idx, mask in enumerate(masks):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)

        m = mask["segmentation"]  # Binary mask
        color = np.random.rand(3)  # Random color for each mask

        # Create a mask image
        masked_image = np.zeros_like(image, dtype=np.float32)
        for i in range(3):  # Apply color to all 3 channels
            masked_image[:, :, i] = m * color[i]

        plt.imshow(masked_image, alpha=0.5)  # Set transparency to make it visible

        # Set the mask number as the title
        plt.title(f"Mask {idx + 1}", fontsize=16, color='white', backgroundcolor='black')

        plt.axis("off")
        plt.show()


def overlay_and_save(image_path, masks, save_path, alpha=0.5):
    """Overlays segmentation masks on the original image and saves it at high resolution."""

    # Load the image using OpenCV (ensuring high resolution)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Load as BGR
    if image is None:
        raise ValueError(f"Error: Could not load image from {image_path}")

    # Convert BGR to RGB for visualization
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create overlay copy
    overlay = image.copy()

    for mask in masks:
        m = mask["segmentation"]  # Extract binary mask
        color = np.random.randint(0, 255, (3,), dtype=np.uint8)  # Random color

        # Apply color to mask
        for i in range(3):  # RGB channels
            overlay[:, :, i] = np.where(m, color[i], overlay[:, :, i])

        # Find and draw contours
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)  # White border

    # Blend original image with mask overlay
    blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

    # Convert back to BGR for OpenCV saving
    blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

    # Save image at high resolution
    #cv2.imwrite(save_path, blended_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # PNG for lossless quality

    print(f"Saved high-resolution image at: {save_path}")
    plt.axis("off")
    plt.imshow(blended)
    plt.show()


def overlay_and_save_binary(image_path, masks, save_path, alpha=0.5):
    """Generates and saves a binary mask based on the provided segmentation masks."""

    # Load the image using OpenCV (ensuring high resolution)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Load as BGR
    if image is None:
        raise ValueError(f"Error: Could not load image from {image_path}")

    # Convert BGR to RGB for visualization
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create an empty binary mask (same size as the image)
    binary_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Iterate over the masks and combine them into the binary mask
    for mask in masks:
        m = mask["segmentation"]  # Extract binary mask
        binary_mask = np.maximum(binary_mask, m.astype(np.uint8))  # Combine masks (OR operation)

    # Save the binary mask as an image (using white for foreground and black for background)
    cv2.imwrite(save_path, binary_mask * 255)  # Multiply by 255 to get 0 (black) and 255 (white)

    print(f"Saved binary mask image at: {save_path}")

    # Visualize the result (optional)
    plt.axis("off")
    plt.imshow(binary_mask, cmap='gray')  # Show in grayscale
    plt.show()

# Load the SAM model
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)

# Initialize mask generator
mask_generator = SamAutomaticMaskGenerator(sam)

# Load and preprocess image
image_path = 'fatbikes/oneperson.png'  # Change this to your image path
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Generate masks

masks = mask_generator.generate(image)
# print(masks)
# print('Original Image ====>')
# print('Predictions ====>')
# show_masks(image, masks)
# print('Predictions ====>')
mask = masks[86]
save_path = 'sam_mask/sam03'
overlay_and_save_binary(image_path, [mask], save_path)


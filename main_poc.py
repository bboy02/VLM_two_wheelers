#Import libraries
import cv2
import time
import psutil
import gc
import matplotlib.pyplot as plt
from torch import nn
from PIL import Image as PILImg
from robokit.ObjDetection import GroundingDINOObjectPredictor, SegmentAnythingPredictor
print("Initialize object detectors")
from utils.img_utils import masks_to_bboxes
from robokit.utils import annotate, overlay_masks
from utils.inference_utils import get_features
from utils.inference_utils import compute_similarity
import numpy as np
import torch
from PIL import Image
import os
import json
from tqdm import trange
import argparse
import math
from utils.inference_utils import FFA_preprocess, get_foreground_mask


def main():
    parser = argparse.ArgumentParser(description='Multi-class Object Detection')
    parser.add_argument('--fatbike_templates', nargs='+', required=True,
                        help='List of fatbike template image paths')
    parser.add_argument('--vespa_templates', nargs='+', required=True,
                        help='List of vespa template image paths')
    parser.add_argument('--cycle_templates', nargs='+', required=True,
                        help='List of cycle template image paths')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for results')
    parser.add_argument('--class_1', required=True,
                        help='First Image Class ')
    parser.add_argument('--class_2', required=True,
                        help='Second Image Class ')
    parser.add_argument('--class_3', required=True,
                        help='Third Image Class ')
    parser.add_argument('--input_dir', required=True,
                        help='Input directory containing test images')
    parser.add_argument('--text_prompt', default='cycles',
                        help='Text prompt for object detection ')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Similarity threshold for classification (default: 0.7)')
    parser.add_argument('--compute_templates', required=True, choices=['True', 'False'],
                        help='Set to "True" to compute templates, "False" to use cached')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save arguments to config file
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    #initializing GroundingDino & SAM

    gdino = GroundingDINOObjectPredictor(use_vitb=False, threshold=0.3) # we set threshold for GroundingDINO here.
    SAM = SegmentAnythingPredictor(vit_model="vit_h")

    #Helper Functions
    def get_bbox_masks_from_gdino_sam(image_path, gdino, SAM, text_prompt='objects', visualize=False):
        """
        Get bounding boxes and masks from gdino and sam
        @param image_path: the image path
        @param gdino: the model of grounding dino
        @param SAM: segment anything model or its variants
        @param text_prompt: generally 'objects' for object detection of noval objects
        @param visualize: if True, visualize the result
        @return: the bounding boxes and masks of the objects.
        Bounding boxes are in the format of [x_min, y_min, x_max, y_max] and shape of (N, 4).
        Masks are in the format of (N, H, W) and the value is True for object and False for background.
        They are both in the format of torch.tensor.
        """
        image_pil = PILImg.open(image_path).convert("RGB")
        print("GDINO: Predict bounding boxes, phrases, and confidence scores")
        with torch.no_grad():
            bboxes, phrases, gdino_conf = gdino.predict(image_pil, text_prompt)
            w, h = image_pil.size  # Get image width and height
            # Scale bounding boxes to match the original image size
            image_pil_bboxes = gdino.bbox_to_scaled_xyxy(bboxes, w, h)
            print(f"image_pil_bboxes: {image_pil_bboxes}")
            if image_pil_bboxes.numel() == 0:  # Checks if the tensor has zero elements
                print("No bounding boxes found!")


            print("SAM prediction")
            image_pil_bboxes, masks = SAM.predict(image_pil, image_pil_bboxes)
        masks = masks.squeeze(1)
        accurate_bboxs = masks_to_bboxes(masks)  # get the accurate bounding boxes from the masks
        accurate_bboxs = torch.tensor(accurate_bboxs)
        bbox_annotated_pil = None
        if visualize:
            print("Annotate the scaled image with bounding boxes, confidence scores, and labels, and display")
            bbox_annotated_pil = annotate(overlay_masks(image_pil, masks), accurate_bboxs, gdino_conf, phrases)
            plt.imshow(bbox_annotated_pil)
            plt.title("bbox annotated")

        return accurate_bboxs, masks, bbox_annotated_pil

    def get_FFA_feature(img_path, mask, encoder, img_size=448):
        """get FFA for a pair of rgb and mask images"""
        # mask_path = img_path.replace('images', 'masks').replace('.jpg', '.png')
        # mask = Image.open(mask_path)
        # mask = mask.convert('L')
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        w, h = img.size

        if (img_size is not None) and (min(w, h) > img_size):
            img.thumbnail((img_size, img_size), Image.LANCZOS)
            mask.thumbnail((img_size, img_size), Image.BILINEAR)

            # mask.show()
        else:
            new_w = math.ceil(w / 14) * 14
            new_h = math.ceil(h / 14) * 14
            img = img.resize((new_w, new_h), Image.LANCZOS)
        # mask = mask.resize((16 , 16), Image.BILINEAR)

        with torch.no_grad():
            preprocessed_imgs = FFA_preprocess([img], img_size).to(device)
            mask_size = img_size // 14
            masks = get_foreground_mask([mask], mask_size).to(device)
            emb = encoder.forward_features(preprocessed_imgs)

            grid = emb["x_norm_patchtokens"].view(1, mask_size, mask_size, -1)
            avg_feature = (grid * masks.permute(0, 2, 3, 1)).sum(dim=(1, 2)) / masks.sum(dim=(1, 2, 3)).unsqueeze(-1)

            return avg_feature

    def get_object_proposal(image_path, bboxs, masks, tag="mask", ratio=1.0, output_dir='object_proposals', save_segm=False, save_proposal=False):
        """
        Get object proposals from the image according to the bounding boxes and masks.

        @param image_path:
        @param bboxs: numpy array, the bounding boxes of the objects [N, 4]
        @param masks: Boolean numpy array of shape [N, H, W], True for object and False for background
        @param tag: use mask or bbox to crop the object
        @param ratio: ratio to resize the image
        @param save_rois: if True, save the cropped object proposals
        @param output_dir: the folder to save the cropped object proposals
        @return: the cropped object proposals and the object proposals information
        """
        raw_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image_height, image_width = raw_image.shape[:-1]
        scene_name = os.path.basename(image_path).split('.')[0]
        sel_rois = []
        rois = []
        cropped_masks = []
        cropped_imgs = []
        # ratio = 0.25
        if ratio != 1.0:
            scene_image = cv2.resize(raw_image, (int(raw_image.shape[1] * ratio), int(raw_image.shape[0] * ratio)),
                                   cv2.INTER_LINEAR)
        else:
            scene_image = raw_image
        for ind in range(len(masks)):
            # bbox
            x0 = int(bboxs[ind][0])
            y0 = int(bboxs[ind][1])
            x1 = int(bboxs[ind][2])
            y1 = int(bboxs[ind][3])

            # load mask
            mask = masks[ind].squeeze(0).cpu().numpy()
            # Assuming `mask` is your boolean numpy array with shape (H, W)
            rle = None
            if save_segm:
                rle = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
                rle['counts'] = rle['counts'].decode('ascii')  # If saving to JSON, ensure counts is a string
            cropped_mask = mask[y0:y1, x0:x1]
            cropped_mask = Image.fromarray(cropped_mask.astype(np.uint8) * 255)
            cropped_masks.append(cropped_mask)
            # show mask
            cropped_img = raw_image[y0:y1, x0:x1]
            cropped_img = Image.fromarray(cropped_img)

            cropped_imgs.append(cropped_img)

            # save bbox
            sel_roi = dict()
            sel_roi['roi_id'] = int(ind)
            sel_roi['mask'] = mask
            #sel_roi['image_id'] = int(scene_name.split('_')[-1])
            sel_roi['bbox'] = [int(x0 * ratio), int(y0 * ratio), int((x1 - x0) * ratio), int((y1 - y0) * ratio)]
            sel_roi['area'] = np.count_nonzero(mask)
            sel_roi['roi_dir'] = os.path.join(output_dir, scene_name, scene_name + '_' + str(ind).zfill(3) + '.png')
            sel_roi['image_dir'] = image_path
            sel_roi['image_width'] = scene_image.shape[1]
            sel_roi['image_height'] = scene_image.shape[0]
            if save_segm:
                sel_roi['segmentation'] = rle  # Add RLE segmentation
            sel_roi['scale'] = int(1 / ratio)
            sel_rois.append(sel_roi)
        if save_proposal:
            with open(os.path.join(output_dir, 'proposals_on_' + scene_name + '.json'), 'w') as f:
                json.dump(sel_rois, f)
        return rois, sel_rois, cropped_imgs, cropped_masks

    def visualize_crops(cropped_imgs, cropped_masks):
        num_objects = len(cropped_imgs)
        fig, axes = plt.subplots(num_objects, 2, figsize=(6, 3 * num_objects))

        if num_objects == 1:
            axes = [axes]  # Ensure axes is iterable for a single image case

        for i in range(num_objects):
            # Convert to numpy array if it's in PIL format
            img = np.array(cropped_imgs[i])
            mask = np.array(cropped_masks[i])

            # Display cropped image
            axes[i][0].imshow(img)
            axes[i][0].set_title(f"Cropped Image {i}")
            axes[i][0].axis("off")

            # Display cropped mask
            axes[i][1].imshow(mask, cmap="gray")
            axes[i][1].set_title(f"Cropped Mask {i}")
            axes[i][1].axis("off")

        plt.tight_layout()
        plt.show()

    def show_mask(mask, ax, color=None, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        elif color is None:
            color = np.array([30/255, 144/255, 255/255, 0.6])  # Default blue color

        # Ensure color is a numpy array for reshaping
        color = np.array(color)

        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def get_cpu_memory_mb():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)  # in MB

    def log_total_usage(start_time, label, output_dir):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - start_time
        cpu_mem = get_cpu_memory_mb()
        gpu_mem = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0

        log_str = f"[{label}] Time: {elapsed:.2f}s | CPU Mem: {cpu_mem:.2f}MB | GPU Mem: {gpu_mem:.2f}MB"
        print(log_str)

        # Log to file
        with open(os.path.join(output_dir, "inference_metrics.log"), "a") as f:
            f.write(log_str + "\n")

        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        gc.collect()

    def show_box(box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

    # Load dino v2
    start_time = time.perf_counter()
    encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg') # define the DINOv2 version
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    encoder.eval()
    img_size = 448

    recompute_templates = args.compute_templates.lower() == 'true'

    # Ensure directory for saved features exists
    os.makedirs('cached_features', exist_ok=True)

    def get_feature_path(class_name):
        """Generate standardized path for saved features"""
        return os.path.join('cached_features', f'{class_name}_features.pt')

    # -----------------------TEMPLATE FEATURE LOADING/SAVING-----------------------#
    def load_or_extract_features(template_paths, class_name, text_prompt):
        """Load features if available, else compute and save them"""
        feat_path = get_feature_path(class_name)

        if not recompute_templates and os.path.exists(feat_path):
            print(f"Loading cached {class_name} features")
            return torch.load(feat_path, map_location=device)

        print(f"Computing {class_name} features...")
        raw_features = extract_template_features(template_paths, text_prompt)
        if raw_features is None:
            return None

        # Normalize and save
        features = nn.functional.normalize(raw_features, dim=1, p=2)
        torch.save(features.cpu(), feat_path)  # Save CPU tensor for device compatibility
        return features

    # -----------------------MULTI-TEMPLATE FEATURE EXTRACTION-----------------------#
    def extract_template_features(template_imgs, text_prompt='objects'):
        """Extracts and aggregates features from multiple template images."""
        all_features = []
        for template_img in template_imgs:
            print(f"Processing template image: {template_img}")
            image = cv2.imread(template_img)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect with Grounding Dino & Segment with SAM
            accurate_bboxs, masks, vis_img = get_bbox_masks_from_gdino_sam(
                template_img, gdino, SAM, text_prompt=text_prompt, visualize=False
            )

            # Check if any objects were detected
            if accurate_bboxs.numel() == 0:
                print(f"No objects detected in {template_img}, skipping.")
                continue

            # Visualization for debugging
            print("Visualizing bounding box and mask...")
            print("bbox shape", accurate_bboxs.shape)
            template_box = accurate_bboxs[0].cpu().numpy()
            template_mask = masks[0].cpu().numpy()
            template_mask = np.expand_dims(template_mask, axis=0)

            # Display the image with mask and bounding box overlay
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            show_mask(template_mask, plt.gca())
            show_box(template_box, plt.gca())
            plt.axis('off')
            plt.title('SAM+GDino')


            # Convert mask to PIL format for feature extraction
            mask_img = (template_mask).astype(np.uint8) * 255  # Convert False to 0, True to 255
            mask_img = Image.fromarray(mask_img.squeeze(), 'L')  # 'L' for (8-bit pixels, black and white)

            # Extract features using DINOv2
            avg_feature = get_FFA_feature(template_img, mask_img, encoder)
            all_features.append(avg_feature)

        if not all_features:
            print("No features extracted from any template image.")
            return None

        # Aggregate features across all template images
        all_features = torch.cat(all_features, dim=0)  # Concatenate along the batch dimension
        combined_features = torch.mean(all_features, dim=0, keepdim=True)  # Average along the batch dimension

        return combined_features


    fatbike_features = load_or_extract_features(args.fatbike_templates, args.class_1, text_prompt=args.text_prompt)
    vespa_features = load_or_extract_features(args.vespa_templates, args.class_2, text_prompt=args.text_prompt)
    cycle_features = load_or_extract_features(args.cycle_templates,  args.class_3,text_prompt=args.text_prompt)


    # Normalize
    if fatbike_features is not None:
        fatbike_features = nn.functional.normalize(fatbike_features, dim=1, p=2)
    if vespa_features is not None:
        vespa_features = nn.functional.normalize(vespa_features, dim=1, p=2)
    if cycle_features is not None:
        cycle_features = nn.functional.normalize(cycle_features, dim=1, p=2)
    else:
        print("No object features extracted. Check template images and detections.")
        # Handle the case where no features were extracted, possibly by exiting or returning.
        exit()
    # -----------------------MULTI-TEMPLATE FEATURE EXTRACTION-----------------------#


    # Directory containing test images
    image_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)


    # Get list of all images in the directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        print(f"Processing {image_file}")


        # Open and convert the test image to RGB
        image_pil = PILImg.open(image_path).convert("RGB")

        # Detect & Segment Test Image
        accurate_bboxs, masks, vis_img = get_bbox_masks_from_gdino_sam(
            image_path, gdino, SAM, text_prompt='cycle', visualize=True
        )

        rois, sel_rois, cropped_imgs, cropped_masks = get_object_proposal(
            image_path, accurate_bboxs, masks, ratio=1.0, output_dir=".", save_proposal=False
        )

        scene_features = []
        for i in trange(len(cropped_imgs)):
            img = cropped_imgs[i]
            mask = cropped_masks[i]
            ffa_feature = get_features([img], [mask], encoder, device=device, img_size=448)
            scene_features.append(ffa_feature)

        scene_features = torch.cat(scene_features, dim=0)
        scene_features = torch.nn.functional.normalize(scene_features, dim=1, p=2)
        print(f"Scene Features for {image_file}: {scene_features.shape}")


        # Compute similarity with both fatbike and vespa features
        sim_fatbike = compute_similarity(fatbike_features, scene_features).squeeze(-1)
        sim_vespa = compute_similarity(vespa_features, scene_features).squeeze(-1)
        sim_cycle = compute_similarity(cycle_features, scene_features).squeeze(-1)

        # Load and convert the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Plot the image
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        # Prepare detection dictionary for this image (for JSON format)
        image_name = os.path.splitext(os.path.basename(image_file))[0]
        detections_json = {
            "image_id": image_name,
            "instances": []
        }

        # Iterate over all scene features
        for idx in range(len(scene_features)):
            score_fatbike = sim_fatbike[idx].item()
            score_vespa = sim_vespa[idx].item()
            score_cycle = sim_cycle[idx].item()

            scores = [score_fatbike, score_vespa, score_cycle]
            max_score = max(scores)
            class_idx = scores.index(max_score)


            # Skip if below threshold
            if max_score < args.threshold:
                continue
            print("info")
            print(max_score, class_idx)
            # Determine class and corresponding color
            if class_idx == 0:  # Fatbike
                predicted_class = "Fatbikes"
                color = (0, 0, 1, 0.5)  # Blue with transparency
            elif class_idx == 1:  # Vespa
                predicted_class = "Vespa"
                color = (0, 1, 0, 0.5)  # Green with transparency
            else:  # Cycle
                predicted_class = "Cycles"
                color = (1, 0, 0, 0.5)  # Red with transparency

            print(f"[{image_file}] Region {idx}: {predicted_class} "
                  f"(Score: {max_score:.3f})")

            # Extract bounding box and mask
            bbox = sel_rois[idx]['bbox']
            mask = sel_rois[idx]['mask']
            x0 = int(bbox[0])
            y0 = int(bbox[1])
            x1 = x0 + int(bbox[2])
            y1 = y0 + int(bbox[3])
            bbox = [x0, y0, x1, y1]


            # Append this detection to JSON structure
            detections_json["instances"].append({
                "label": predicted_class,
                "bbox": bbox,
                "score": max_score
            })

            # Save per-image JSON file
            json_path = os.path.join(output_dir, f"{image_name}.json")
            with open(json_path, 'w') as f:
                json.dump(detections_json, f, indent=2)

            print(f"Saved JSON-format detection results for {image_file} to {json_path}")

            # Show mask with class-specific color
            show_mask(mask, plt.gca(), color=color)
            show_box(bbox, plt.gca())

        # Add legend to indicate class colors
        import matplotlib.patches as mpatches

        legend_patches = [
            mpatches.Patch(color=(0, 1, 0, 0.5), label='Vespa'),  # Green
            mpatches.Patch(color=(0, 0, 1, 0.5), label='Fatbike'),  # Blue
            mpatches.Patch(color=(1, 0, 0, 0.5), label='Cycle')  # Red
        ]
        plt.legend(handles=legend_patches, loc='lower left', fontsize='large', frameon=True)

        plt.axis('off')

        # Save the final visualization to the output folder
        output_image_path = os.path.join(output_dir, f"{image_name}_prediction.png")
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
        print(f"Saved prediction visualization for {image_file} to {output_image_path}")
        log_total_usage(start_time, os.path.basename(image_path), args.output_dir)
        plt.show()





if __name__ == "__main__":
    main()











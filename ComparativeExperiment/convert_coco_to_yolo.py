import json
import os
import shutil
from tqdm import tqdm

def convert_coco_to_yolo_segmentation(json_file, output_dir, image_dir):
    with open(json_file) as f:
        data = json.load(f)

    images = {img['id']: img for img in data['images']}
    categories = {cat['id']: cat for cat in data['categories']}
    
    # Create output directories
    labels_dir = os.path.join(output_dir, 'labels')
    images_output_dir = os.path.join(output_dir, 'images')
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_output_dir, exist_ok=True)

    # Group annotations by image_id
    img_anns = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_anns:
            img_anns[img_id] = []
        img_anns[img_id].append(ann)

    for img_id, anns in tqdm(img_anns.items()):
        img_info = images[img_id]
        file_name = img_info['file_name']
        width = img_info['width']
        height = img_info['height']
        
        # Copy image
        src_img_path = os.path.join(image_dir, file_name)
        dst_img_path = os.path.join(images_output_dir, file_name)
        if os.path.exists(src_img_path):
            shutil.copy(src_img_path, dst_img_path)
        else:
            print(f"Warning: Image {src_img_path} not found.")
            continue

        # Create label file
        label_file = os.path.splitext(file_name)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        
        with open(label_path, 'w') as f:
            for ann in anns:
                cat_id = ann['category_id']
                # Map category id if necessary. Assuming 0 for single class or using original id.
                # In YOLO, classes are 0-indexed.
                # In train.json, we saw category_id 1. We should map it to 0.
                class_id = 0 # Assuming single class 'tongue'
                
                segmentation = ann['segmentation']
                if isinstance(segmentation, list):
                    for seg in segmentation:
                        # seg is [x1, y1, x2, y2, ...]
                        # Normalize
                        normalized_seg = []
                        for i in range(0, len(seg), 2):
                            x = seg[i] / width
                            y = seg[i+1] / height
                            normalized_seg.append(f"{x:.6f} {y:.6f}")
                        
                        line = f"{class_id} {' '.join(normalized_seg)}\n"
                        f.write(line)

if __name__ == "__main__":
    DATA_DIR = "/data/projects/tongue_segmentation/sam2/dataset"
    IMAGES_DIR = os.path.join(DATA_DIR, "image")
    MASKS_FILE = os.path.join(DATA_DIR, "dataset.json")
    OUTPUT_DIR = "/data/projects/tongue_segmentation/ComparativeExperiment/yolo_dataset"
    
    convert_coco_to_yolo_segmentation(MASKS_FILE, OUTPUT_DIR, IMAGES_DIR)

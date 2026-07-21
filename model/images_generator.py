import cv2
import os
import albumentations as A

# ----------- INPUTS --------------
image_path = "model/Apple_Datasets/train/images/frame_0.jpg"
label_path = "model/Apple_Datasets/train/labels/frame_0.txt"   # YOLO format
output_dir_images = "model/Apple_Datasets/train/images"
output_dir_labels = "model/Apple_Datasets/train/labels"
num_images = 600
# ---------------------------------

# Load original image
image = cv2.imread(image_path)
h, w = image.shape[:2]

# Load YOLO annotation
with open(label_path, "r") as f:
    yolo_data = f.readlines()

bboxes = []
classes = []

for line in yolo_data:
    cls, x, y, bw, bh = map(float, line.split())
    classes.append(int(cls))
    bboxes.append([x, y, bw, bh])

# Augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.Rotate(limit=20, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(p=0.3),
    A.MotionBlur(p=0.3),
    A.HueSaturationValue(p=0.5),
    A.RandomGamma(p=0.4),
    A.Perspective(p=0.4),
    A.RandomScale(scale_limit=0.2, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2, rotate_limit=15, p=0.7)
], bbox_params=A.BboxParams(format='yolo', label_fields=["category_ids"]))

# Generate images
for i in range(num_images):
    print(i)
    augmented = transform(image=image, bboxes=bboxes, category_ids=classes)
    aug_img = augmented["image"]
    aug_bboxes = augmented["bboxes"]
    
    img_name = f"frame_{i+1}.jpg"
    txt_name = f"frame_{i+1}.txt"

    cv2.imwrite(os.path.join(output_dir_images, img_name), aug_img)

    # Save updated YOLO label
    with open(os.path.join(output_dir_labels, txt_name), "w") as f:
        for cls, box in zip(classes, aug_bboxes):
            f.write(f"{cls} {' '.join(map(str, box))}\n")

print("600 images generated successfully!")

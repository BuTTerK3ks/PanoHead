import os
import torch
from PIL import Image
from torchvision.transforms import ToPILImage
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
from torchvision import transforms


def get_mask(model, batch, cid):
    normalized_batch = transforms.functional.normalize(batch, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    output = model(normalized_batch)['out']
    normalized_masks = torch.nn.functional.softmax(output, dim=1)
    boolean_car_masks = (normalized_masks.argmax(1) == cid)
    return boolean_car_masks.float()


# Define the preprocessing transformation
preprocess = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# load segmentation net
seg_net = deeplabv3_resnet101(pretrained=True, progress=False).to('cuda:0')
seg_net.requires_grad_(False)
seg_net.eval()


'''
# load segmentation net with updated weights parameter
seg_net = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT).to('cuda:0')
seg_net.requires_grad_(False)
seg_net.eval()
'''

# Directory containing the full images
full_images_dir = 'images_cropped'
# Directory to save the segmented images
segmented_images_dir = './images_segmented'

# Create the segmented_images_dir if it doesn't exist
os.makedirs(segmented_images_dir, exist_ok=True)

# Iterate over each file in the full_images_dir
for filename in os.listdir(full_images_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Check for image files
        file_path = os.path.join(full_images_dir, filename)

        # Open and preprocess the image
        image = Image.open(file_path).convert('RGB')  # Convert image to RGB format
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0).to('cuda:0')

        # 15 means human mask
        mask0 = get_mask(seg_net, input_batch, 15).unsqueeze(1)

        # Squeeze the tensor to remove unnecessary dimensions and convert to PIL Image
        mask_squeezed = torch.squeeze(mask0)
        mask_image = ToPILImage()(mask_squeezed)

        # Remove the suffix from the filename and add a new one for the segmented image
        new_filename = os.path.splitext(filename)[0] + '.png'
        # Save as PNG in the segmented_images_dir
        mask_image.save(os.path.join(segmented_images_dir, new_filename))

import torch
import torchvision
from PIL import Image
import numpy as np
from carvekit.api.high import HiInterface

# Load MiDaS model for depth estimation

import torch
from PIL import Image
import numpy as np

class DepthFeatureExtractor(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        # Load MiDaS model
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device)
        # MiDaS transforms
        self.transforms = torch.hub.load("intel-isl/MiDaS", "transforms").default_transform

    def forward(self, img_list):
        depth_list = []
        for img in img_list:
            # Convert PIL image to NumPy array for MiDaS
            if isinstance(img, Image.Image):
                img = np.array(img)

            # Convert RGBA to RGB if necessary
            if img.shape[-1] == 4:  # Check if it has 4 channels (RGBA)
                img = img[..., :3]  # Discard the alpha channel

            # Transform the image for depth estimation
            transformed = self.transforms(img)
            print(f"Transformed data type: {type(transformed)}")  # Debugging: print the type of transformed data
            
            # If transformed is a tensor, we should handle it directly
            if isinstance(transformed, torch.Tensor):
                img_input = transformed.to(next(self.midas.parameters()).device)  # Use the device of the model
            else:
                # Check if it's a dict with an "image" key
                if isinstance(transformed, dict) and "image" in transformed:
                    img_input = transformed["image"].to(next(self.midas.parameters()).device)
                else:
                    raise ValueError("Transformed output is not a tensor or lacks 'image' key.")

            # Debugging: Print the shape of img_input
            print(f"Shape of transformed input: {img_input.shape}")

            # Ensure the input has 4 dimensions for MiDaS
            if img_input.dim() == 3:  # If 3D, add a batch dimension
                img_input = img_input.unsqueeze(0)

            with torch.no_grad():
                depth = self.midas(img_input)

            # If MiDaS returns a 4D tensor, squeeze the batch dimension before appending to depth_list
            if depth.dim() == 4:
                depth = depth.squeeze(0)  # Remove batch dimension
            
            depth_list.append(depth)
        
        return torch.stack(depth_list, dim=0)

# Foreground Feature Averaging using DINOv2 and MiDaS
class ForegroundFeatureAveraging(torch.nn.Module):
    def __init__(self, device, carvekit_object_type="object"):
        super().__init__()
        self.device = device
        # Load DINOv2 model
        self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(self.device)

        # Load MiDaS model for depth estimation
        self.depth_extractor = DepthFeatureExtractor(self.device)

        # CarveKit interface for foreground segmentation
        self.interface = HiInterface(object_type=carvekit_object_type,  
                                     batch_size_seg=5,
                                     batch_size_matting=1,
                                     device=str(self.device),
                                     seg_mask_size=640, 
                                     matting_mask_size=2048,
                                     trimap_prob_threshold=231,
                                     trimap_dilation=30,
                                     trimap_erosion_iters=5,
                                     fp16=False)

    def preprocess(self, x_list):
        preprocessed_images = []
        for x in x_list:
            def _to_rgb(img):
                if img.mode != "RGB":
                    img = img.convert("RGB")
                return img

            preprocessed_image = torchvision.transforms.Compose([
                _to_rgb,
                torchvision.transforms.Resize((336, 336), interpolation=Image.BICUBIC),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])(x)
            preprocessed_images.append(preprocessed_image)

        return torch.stack(preprocessed_images, dim=0).to(self.device)

    def get_foreground_mask(self, tensor_imgs):
        masks = []
        for tensor_img in tensor_imgs:
            tensor_img = tensor_img.detach().cpu()
            numpy_img_sum = tensor_img.sum(dim=0).numpy()
            min_value = np.min(numpy_img_sum)
            mask = ~(numpy_img_sum == min_value)
            mask = mask.astype(np.uint8)
            mask = Image.fromarray(mask * 255)
            resized_mask = mask.resize((24, 24), Image.BILINEAR)
            resized_mask_numpy = np.array(resized_mask)
            resized_mask_numpy = resized_mask_numpy / 255.0
            tensor_mask = torch.from_numpy(resized_mask_numpy.astype(np.float32))
            tensor_mask[tensor_mask > 0.5] = 1.0
            tensor_mask = tensor_mask.unsqueeze(0).long().to(self.device)
            if tensor_mask.sum() == 0:
                tensor_mask = torch.ones_like(tensor_mask)
            masks.append(tensor_mask)
        return torch.stack(masks, dim=0)

    def forward(self, variant, *x):
        """
        :param variant: either "Crop-Feat" or "Crop-Img".
        :param x: Input image(s).
        :return: Processed feature vectors or similarity score.
        """
        if len(x) == 1 and (isinstance(x[0], list) or isinstance(x[0], torch.Tensor)):
            return self.forward_single(x[0], variant)
        elif len(x) == 1:
            return self.forward_single([x[0]], variant)
        elif len(x) == 2:
            return torch.cosine_similarity(self.forward_single(x[0], variant)[0], self.forward_single(x[1], variant)[0], dim=0).cpu().item()
        else:
            raise ValueError("Invalid number of inputs, only 1 or 2 inputs are supported.")

    def forward_single(self, x_list, variant):
        with torch.no_grad():
            # Apply foreground segmentation
            img_list = [np.array(self.interface([x])[0]) for x in x_list]
            for img in img_list:
                img[img[..., 3] == 0] = [0, 0, 0, 0]
            img_list = [Image.fromarray(img) for img in img_list]
            preprocessed_imgs = self.preprocess(img_list)

            # Get foreground masks
            masks = self.get_foreground_mask(preprocessed_imgs)

            # DINOv2 embeddings
            dino_emb = self.encoder.forward_features(preprocessed_imgs)

            # Depth embeddings from MiDaS
            depth_imgs = self.depth_extractor(img_list)

            # Handle different variants (Crop-Feat or Crop-Img)
            if variant == "Crop-Feat":
                grid = dino_emb["x_norm_patchtokens"].view(len(x_list), 24, 24, -1)
            elif variant == "Crop-Img":
                grid = self.encoder.forward_features(self.preprocess(x_list))["x_norm_patchtokens"].view(len(x_list), 24, 24, -1)
            else:
                raise ValueError("Invalid variant, only Crop-Feat and Crop-Img are supported.")

            # Combine DINOv2, Depth, and Foreground mask
            combined_embedding = (grid * masks.permute(0, 2, 3, 1)).sum(dim=(1, 2)) / masks.sum(dim=(1, 2, 3)).unsqueeze(-1)

            # Add the depth features (downsampling the depth to match the grid)
            depth_resized = torch.nn.functional.interpolate(depth_imgs, size=(24, 24), mode='bilinear', align_corners=False)
            depth_emb = depth_resized.view(len(x_list), -1)

            return torch.cat([combined_embedding, depth_emb], dim=-1)

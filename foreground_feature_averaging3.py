import torch
import torchvision
from PIL import Image
import numpy as np
from transformers import DPTFeatureExtractor, DPTForDepthEstimation

class ForegroundFeatureAveraging(torch.nn.Module):
    def __init__(self, device):
        """
        :param device: string or torch.device object to run the model on.
        """
        super().__init__()
        self.device = device
        self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.encoder.to(self.device)
        
        # MiDaS model for depth estimation
        self.midas = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(self.device)
        self.feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")

    def preprocess(self, x_list):
        preprocessed_images = []

        for x in x_list:
            # Ensure the image is in RGB format
            def _to_rgb(x):
                if x.mode != "RGB":
                    x = x.convert("RGB")
                return x

            preprocessed_image = torchvision.transforms.Compose([
                _to_rgb,
                torchvision.transforms.Resize((336, 336), interpolation=Image.BICUBIC),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])(x)
            preprocessed_images.append(preprocessed_image)

        return torch.stack(preprocessed_images, dim=0).to(self.device)

    def get_foreground_mask(self, img_list):
        masks = []

        # Preprocess images for MiDaS depth estimation
        inputs = self.feature_extractor(images=img_list, return_tensors="pt").to(self.device)
        
        # Get depth maps from MiDaS
        with torch.no_grad():
            depth_maps = self.midas(**inputs).predicted_depth.squeeze(1)
        
        # Normalize depth maps to 0-1 range
        depth_maps = (depth_maps - depth_maps.min()) / (depth_maps.max() - depth_maps.min())

        # Generate binary foreground mask based on depth
        for depth_map in depth_maps:
            depth_map = depth_map.cpu().numpy()
            # Apply a threshold to distinguish foreground from background
            mask = depth_map > 0.5
            mask = mask.astype(np.uint8) * 255
            mask = Image.fromarray(mask)
            resized_mask = mask.resize((24, 24), Image.BILINEAR)
            resized_mask_numpy = np.array(resized_mask) / 255.0
            tensor_mask = torch.from_numpy(resized_mask_numpy.astype(np.float32))
            tensor_mask[tensor_mask > 0.5] = 1.0
            tensor_mask = tensor_mask.unsqueeze(0).long().to(self.device)
            if tensor_mask.sum() == 0:
                tensor_mask = torch.ones_like(tensor_mask)
            masks.append(tensor_mask)

        return torch.stack(masks, dim=0)

    def forward(self, variant, *x):
        """
        :param variant: either "Crop-Feat" or "Crop-Img". This determines whether foreground cropping is applied directly
        to the features ("Crop-Feat") or the images ("Crop-Img").
        :param x: Either (1) a single list/tensor of images, or (2) a single image, or (3) two lists of images for comparison.
        :return: If (1) or (2), the computed feature vectors for each image.
        If (3), the cosine similarity between the two sets of feature vectors.
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
            preprocessed_imgs = self.preprocess(x_list)
            masks = self.get_foreground_mask(x_list)

            if variant == "Crop-Feat":
                emb = self.encoder.forward_features(preprocessed_imgs)
            elif variant == "Crop-Img":
                emb = self.encoder.forward_features(self.preprocess(x_list))
            else:
                raise ValueError("Invalid variant, only Crop-Feat and Crop-Img are supported.")

            grid = emb["x_norm_patchtokens"].view(len(x_list), 24, 24, -1)
            return (grid * masks.permute(0, 2, 3, 1)).sum(dim=(1, 2)) / masks.sum(dim=(1, 2, 3)).unsqueeze(-1)

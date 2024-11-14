# model.py
import torch
import torchvision
from PIL import Image
import numpy as np
from carvekit.api.high import HiInterface

# Load MiDaS model for depth estimation
class DepthFeatureExtractor(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device)
        self.transforms = torch.hub.load("intel-isl/MiDaS", "transforms").default_transform

    def forward(self, img_list):
        depth_list = []
        for img in img_list:
            if isinstance(img, Image.Image):
                img = np.array(img)
            if img.shape[-1] == 4:  # RGBA to RGB
                img = img[..., :3]
            transformed = self.transforms(img)
            img_input = transformed.to(next(self.midas.parameters()).device)
            if img_input.dim() == 3:
                img_input = img_input.unsqueeze(0)

            with torch.no_grad():
                depth = self.midas(img_input)
            depth = depth.squeeze(0)
            depth_list.append(depth)
        return torch.stack(depth_list, dim=0)

class AttentionModule(torch.nn.Module):
    def __init__(self, embed_dim):
        super(AttentionModule, self).__init__()
        self.attn_layer = torch.nn.Sequential(
            torch.nn.Linear(embed_dim * 3, embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim, 3),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, dino_emb, depth_emb, foreground_emb):
        combined_emb = torch.cat((dino_emb, depth_emb, foreground_emb), dim=-1)
        attention_weights = self.attn_layer(combined_emb)
        return attention_weights

class ForegroundFeatureAveragingWithAttention(torch.nn.Module):
    def __init__(self, device, carvekit_object_type="object"):
        super().__init__()
        self.device = device
        self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(self.device)
        self.depth_extractor = DepthFeatureExtractor(self.device)
        self.interface = HiInterface(object_type=carvekit_object_type,
                                     batch_size_seg=5, batch_size_matting=1,
                                     device=str(self.device),
                                     seg_mask_size=640, matting_mask_size=2048)
        self.attention_module = AttentionModule(embed_dim=768)

    def preprocess(self, x_list):
        preprocessed_images = []
        for x in x_list:
            preprocessed_image = torchvision.transforms.Compose([
                torchvision.transforms.Resize((336, 336), interpolation=Image.BICUBIC),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])(x)
            preprocessed_images.append(preprocessed_image)
        return torch.stack(preprocessed_images, dim=0).to(self.device)

    def get_foreground_mask(self, tensor_imgs):
        masks = []
        for tensor_img in tensor_imgs:
            numpy_img_sum = tensor_img.sum(dim=0).numpy()
            mask = ~(numpy_img_sum == np.min(numpy_img_sum)).astype(np.uint8)
            resized_mask = Image.fromarray(mask * 255).resize((24, 24), Image.BILINEAR)
            tensor_mask = torch.from_numpy(np.array(resized_mask) / 255.0).unsqueeze(0).long().to(self.device)
            masks.append(tensor_mask)
        return torch.stack(masks, dim=0)

    def forward(self, x_list):
        img_list = [np.array(self.interface([x])[0]) for x in x_list]
        preprocessed_imgs = self.preprocess([Image.fromarray(img) for img in img_list])
        masks = self.get_foreground_mask(preprocessed_imgs)

        dino_emb = self.encoder.forward_features(preprocessed_imgs)
        depth_imgs = self.depth_extractor([Image.fromarray(img) for img in img_list])
        depth_resized = torch.nn.functional.interpolate(depth_imgs, size=(24, 24), mode='bilinear', align_corners=False)
        depth_emb = depth_resized.view(len(x_list), -1)
        
        grid = dino_emb["x_norm_patchtokens"].view(len(x_list), 24, 24, -1)
        combined_embedding = (grid * masks.permute(0, 2, 3, 1)).sum(dim=(1, 2)) / masks.sum(dim=(1, 2, 3)).unsqueeze(-1)
        
        attention_weights = self.attention_module(dino_emb, depth_emb, combined_embedding)
        weighted_embedding = (attention_weights[:, 0].unsqueeze(-1) * dino_emb) + \
                             (attention_weights[:, 1].unsqueeze(-1) * depth_emb) + \
                             (attention_weights[:, 2].unsqueeze(-1) * combined_embedding)
        return weighted_embedding

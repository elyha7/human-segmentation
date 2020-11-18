import torch
import numpy as np
import cv2
from utils.json_processing import get_masks, create_mask
from PIL import Image
import albumentations as albu
import segmentation_models_pytorch as smp
import sys


class PersonSegmenter():
    def __init__(self, weights_path,device='cpu'):
        self.device = device
        self.model,self.preprocessing_fn = self.load_model(weights_path)
        

    def load_model(self, weights_path):
        ENCODER = 'resnet50'
        ENCODER_WEIGHTS = 'imagenet'
        ACTIVATION = 'sigmoid'
        DEVICE = self.device
        model = smp.Unet(
            encoder_name=ENCODER, 
            encoder_weights=None, 
            activation=ACTIVATION,
        )
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

        return model,preprocessing_fn

    def get_validation_augmentation(self, img_size, target_size=480):
        """
            Validation augmentation.
        """
        img_size = min(img_size[:2])
        inter_method = cv2.INTER_AREA if img_size > target_size else cv2.INTER_CUBIC
        test_transform = [
            albu.Resize(target_size, target_size,interpolation=inter_method),
            albu.Lambda(image=self.preprocessing_fn),
            albu.Lambda(image=self.to_tensor),
        ]
        return albu.Compose(test_transform)

    def to_tensor(self,x, **kwargs):
        """
            Transform numpy array to torch tensor.
        """
        return x.transpose(2, 0, 1).astype('float32')

    def preprocess_image(self,img_name):
        """
            Transform image into torch tensor with shape (3,320,320)
        """
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transform = self.get_validation_augmentation(img.shape)
        transformed_img = transform(image=img)['image']
        x_tensor = torch.from_numpy(transformed_img).to(self.device).unsqueeze(0)
        return x_tensor, img

    def get_mask(self,image_tensor, image_numpy, model):
        """
            Predict mask with segmentation model.
        """
        pr_mask = model.predict(image_tensor)
        pr_mask = pr_mask.squeeze().cpu().numpy().round()
        mask = cv2.resize(
            pr_mask, image_numpy.shape[:2][::-1], cv2.INTER_CUBIC)
        return mask

    def visualize_layered(self,image, mask):
        """
            Draw mask on a top of the original image.
        """
        img = np.copy(image)
        img[..., 0] = np.clip(img[..., 0] + mask*64, 0, 255).astype('uint8')
        return img

    def get_bokeh(self,image, mask, kernel_size=30):
        """
            Get image with bokeh effect.
        """
        inverted_mask = 1 - mask
        image_person = np.stack(
            [image[..., i]*mask for i in range(3)], axis=-1).astype('uint8')
        blured_img = cv2.blur(image, (kernel_size, kernel_size))
        without_person = np.stack(
            [blured_img[..., i]*inverted_mask for i in range(3)], axis=-1).astype('uint8')
        final_img = without_person + image_person
        return final_img
    def get_bw(self,image,mask):
        """
            Get image with black and white effect.
        """
        inverted_mask = 1 - mask
        image_person = np.stack([image[...,i]*mask for i in range(3)],axis=-1).astype('uint8')
        bw_img = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),cv2.COLOR_GRAY2RGB)
        without_person = np.stack([bw_img[...,i]*inverted_mask for i in range(3)],axis=-1).astype('uint8')
        final_img = without_person + image_person
        return final_img
    
    def __call__(self, img_path, trans_type, blur_power, result_name='test'):
        img_tensor,img_numpy = self.preprocess_image(img_path)
        print(img_numpy.shape)
        # SEGMENTATION MASK
        mask = self.get_mask(img_tensor,img_numpy,self.model)
        if trans_type == 'bokeh':
            bokeh_img = self.get_bokeh(img_numpy,mask,kernel_size=max(img_numpy.shape)//150*blur_power)
            Image.fromarray(bokeh_img).save(result_name)
        if trans_type == 'layered':
            layered_img = self.visualize_layered(img_numpy, mask)
            Image.fromarray(layered_img).save(result_name)
        if trans_type == 'bnw':
            bnw_img = self.get_bw(img_numpy,mask)
            Image.fromarray(bnw_img).save(result_name)     
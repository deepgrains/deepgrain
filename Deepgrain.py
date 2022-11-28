from PIL import Image, ImageOps
import torchvision.transforms as tf
import torchvision.models.segmentation
import torch
from keras import backend as K
import cv2
import os
import matplotlib.pyplot as plt
import ttach as tta
from torchmetrics import Dice
import numpy as np

class Deepgrain:
    def load_model(self, model_path="model.torch"):      

        # if not os.path.exists(modelPath):      

        Net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)  
        Net.classifier[4] = torch.nn.Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1)) 
        Net = Net.to(self.device)
        Net.load_state_dict(torch.load(model_path))
        Net.eval(); 
    
        return Net
    
    def __init__(self):
        self.height = 1300
        self.width = 1300
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
        self.Net = self.load_model()
        self.transformImg = tf.Compose([tf.ToPILImage(), tf.Resize((self.height, self.width)), tf.ToTensor()])  
        self.transformImgResizeToTensor = tf.Compose([tf.ToPILImage(), tf.Resize((self.height, self.width)), tf.ToTensor()]) 
        self.transformImgToTensor = tf.Compose([tf.PILToTensor()])
        
    def infer_for_segmentations(self, Img):    
        if(len(Img.size())==2):
            Img = torch.unsqueeze(Img, 0)
            Img = Img.repeat(3, 1, 1)    
        height_orgin = Img.size()[1]
        widh_orgin = Img.size()[2]
            
        Img = self.transformImgResizeToTensor(Img)
        Img = torch.autograd.Variable(Img, requires_grad=False).to(self.device).unsqueeze(0)

        with torch.no_grad():
            Prd = self.Net(Img)['out']    
        Prd = tf.Resize((height_orgin,widh_orgin))(Prd[0])
        seg = torch.argmax(Prd, 0).cpu().detach().numpy()      
        return seg
    
    def infer(self, image_file_path):   
        Img = Image.open(image_file_path)
        Img = self.transformImgToTensor(Img)        

        seg = self.infer_for_segmentations(Img).astype(float)  

        grain = np.copy(seg)
        mechanical = np.copy(seg)

        if mechanical is not None:  mechanical[ seg  == 2 ] = 1
        if mechanical is not None:  mechanical[ seg  == 1 ] = 0

        if grain is not None:  grain[ seg == 2 ] = 1      

        return grain, mechanical

    def infer_for_segmentations_tta(self, Img):    
        if(len(Img.size())==2):
            Img = torch.unsqueeze(Img, 0)
            Img = Img.repeat(3, 1, 1)    
        height_orgin = Img.size()[1]
        widh_orgin = Img.size()[2]

        Img = self.transformImgResizeToTensor(Img)
        Img = torch.autograd.Variable(Img, requires_grad=False).to(self.device).unsqueeze(0)

        with torch.no_grad():
            Prd = self.Net(Img)['out']    
        Prd = tf.Resize((height_orgin,widh_orgin))(Prd[0])
        seg = torch.argmax(Prd, 0).cpu().detach().numpy()      
        return seg

    def meanMask(self, nd_array_list):
        nd_sum = nd_array_list[0]
        for nd_array in nd_array_list[1:]:
            nd_sum = sum([nd_sum, nd_array])    
        return nd_sum/len(nd_array_list)

    def get_num_of_pixels(self, gt, annotation):
        summed_matrix = sum([gt, annotation])
        summed_matrix[gt == 0 ] = 0
        summed_matrix[summed_matrix == 1] = 0

        return np.count_nonzero(summed_matrix == 2)

    # Inspired by https://github.com/qubvel/ttach
    def tta_infer(self, image_file_path):
        transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.Rotate90(angles=[0, 180])        
        ])

        Img = Image.open(image_file_path)
        Img = self.transformImgToTensor(Img)

        Img = torch.unsqueeze(Img, 0)    

        masks_grain = []
        masks_mechanical = []

        for one_transformer in transforms:      
            augmented_image = one_transformer.augment_image(Img)
            augmented_image = torch.squeeze(augmented_image)

            seg = self.infer_for_segmentations_tta(augmented_image).astype(float)    

            AnnMap = np.zeros(Img.shape[0:2],np.float32)

            grain = np.copy(seg)
            mechanical = np.copy(seg)

            if mechanical is not None:  mechanical[ seg  == 2 ] = 1
            if mechanical is not None:  mechanical[ seg  == 1 ] = 0

            if grain is not None:  grain[ seg == 2 ] = 1        

            grain = torch.from_numpy(grain)
            mechanical = torch.from_numpy(mechanical) 

            grain = torch.unsqueeze(grain,0)
            grain = torch.unsqueeze(grain,0)

            mechanical = torch.unsqueeze(mechanical,0)
            mechanical = torch.unsqueeze(mechanical,0)

            deaug_mask_grain = one_transformer.deaugment_mask(grain)     
            deaug_mask_mechanical = one_transformer.deaugment_mask(mechanical)     

            deaug_mask_grain = torch.squeeze(deaug_mask_grain)        
            deaug_mask_mechanical = torch.squeeze(deaug_mask_mechanical)        

            masks_grain.append(deaug_mask_grain.cpu().detach().numpy())
            masks_mechanical.append(deaug_mask_mechanical.cpu().detach().numpy())

        mask_mechanical = self.meanMask(masks_mechanical)
        mask_grain = self.meanMask(masks_grain)
        mask_grain[mask_grain >= 0.5 ] = 1
        mask_grain[mask_grain < 0.5 ] = 0

        mask_mechanical[mask_mechanical >= 0.5 ] = 1
        mask_mechanical[mask_mechanical < 0.5 ] = 0

        return mask_grain, mask_mechanical

    def generateVisualization(self, image_path, mechanical_mask, grain_mask, mechanical_count, show_visualizations=True, detailed_visualization=True, save_visualizations=None):    
        if save_visualizations:
            if not os.path.exists(os.path.join("visualizations/")):      
                os.makedirs(os.path.join("visualizations/"))

        if detailed_visualization:
            plt.figure()
            f, axarr = plt.subplots(4,2, figsize=(25,25))
            [axi.set_axis_off() for axi in axarr.ravel()]

            grain_img = cv2.imread(image_path)
            mask = np.stack((mechanical_mask,)*3, axis=-1)
            masked_img = np.copy(grain_img)
            masked_img[(mask==1.0).all(-1)] = [0, 0, 255]

            masked_opacity = cv2.addWeighted(masked_img, 0.3, grain_img, 0.7, 0, masked_img)    

            grain_img = cv2.imread(image_path)
            mask = np.stack((grain_mask,)*3, axis=-1)   
            masked_img = np.copy(grain_img)
            masked_img[(mask==1.0).all(-1)] = [0, 0, 255]    

            masked_opacity_grain = cv2.addWeighted(masked_img, 0.3, grain_img, 0.7, 0, masked_img)    

            grain_img = cv2.imread(image_path)
            mask = np.stack((mechanical_mask,)*3, axis=-1)
            mask_grain = np.stack((grain_mask,)*3, axis=-1)
            masked_img_grain = np.copy(grain_img)
            masked_img = np.copy(grain_img)
            masked_img[(mask==1.0).all(-1)] = [0, 0, 255]
            masked_img_grain[(mask_grain==1.0).all(-1)] = [0, 255, 0]

            masked_opacity_grain_mechanical = cv2.addWeighted(masked_img_grain, 0.3, grain_img, 0.7, 0, masked_img_grain)
            masked_opacity_grain_mechanical = cv2.addWeighted(masked_img, 0.3, masked_opacity_grain_mechanical, 0.7, 0, masked_img)

            grain_img = cv2.imread(image_path)
            grain_img[(grain_mask==0.0)] = 0
            mask = np.stack((mechanical_mask,)*3, axis=-1)
            mask_grain = np.stack((grain_mask,)*3, axis=-1)
            masked_img_grain = np.copy(grain_img)
            masked_img = np.copy(grain_img)
            masked_img[(mask==1.0).all(-1)] = [0, 0, 255]
            masked_img_grain[(mask_grain==1.0).all(-1)] = [0, 255, 0]

            masked_opacity_grain_mechanical_no_bg = cv2.addWeighted(masked_img_grain, 0.3, grain_img, 0.7, 0, masked_img_grain)
            masked_opacity_grain_mechanical_no_bg = cv2.addWeighted(masked_img, 0.3, masked_opacity_grain_mechanical_no_bg, 0.7, 0, masked_img)

            axarr[0,0].imshow(cv2.imread(image_path))

            axarr[1,1].imshow(mechanical_mask)
            axarr[1,0].imshow(grain_mask)
            axarr[2,0].imshow(masked_opacity)
            axarr[2,1].imshow(masked_opacity_grain)
            axarr[3,0].imshow(masked_opacity_grain_mechanical)
            axarr[0,1].imshow(masked_opacity_grain_mechanical_no_bg)

            axarr[0,0].title.set_text('Grain image')
            axarr[0,1].title.set_text('Segmentation')

            axarr[1,0].title.set_text('Grain segmentation')
            axarr[1,1].title.set_text('Mechanical marks')

            axarr[2,0].title.set_text('Grain segmentation')
            axarr[2,1].title.set_text('Mechanical marks')

            axarr[3,0].title.set_text('Mechanical joined together')

            plt.suptitle('Grain segmentation, mechanical marks: ' + str(round(mechanical_count)) + "%") 

            if save_visualizations:
                plt.savefig(os.path.join("visualizations/" + image_path), dpi=300)

            if not show_visualizations:            
                plt.close()
        else:
            grain_img = cv2.imread(image_path)
            grain_img[(grain_mask==0.0)] = 0
            mask = np.stack((mechanical_mask,)*3, axis=-1)
            mask_grain = np.stack((grain_mask,)*3, axis=-1)
            masked_img_grain = np.copy(grain_img)
            masked_img = np.copy(grain_img)
            masked_img[(mask==1.0).all(-1)] = [0, 0, 255]
            masked_img_grain[(mask_grain==1.0).all(-1)] = [0, 255, 0]

            masked_opacity_grain_mechanical_no_bg = cv2.addWeighted(masked_img_grain, 0.3, grain_img, 0.7, 0, masked_img_grain)
            masked_opacity_grain_mechanical_no_bg = cv2.addWeighted(masked_img, 0.3, masked_opacity_grain_mechanical_no_bg, 0.7, 0, masked_img)        

            cv2.imwrite(os.path.join("visualizations/" + image_path), masked_opacity_grain_mechanical_no_bg)
            if show_visualizations:
                plt.imshow(masked_opacity_grain_mechanical_no_bg)

    def count_mechanical(self, mechanical_mask, grain_mask):    
        n_pixels_semantic_grain = np.count_nonzero(grain_mask == 1)
        n_pixels_mechanical_mask = self.get_num_of_pixels(grain_mask, mechanical_mask)
        p_of_mechanical_mask = 0

        if n_pixels_semantic_grain > 0.0:
            if n_pixels_mechanical_mask > 0.0:
                p_of_mechanical_mask = n_pixels_mechanical_mask/n_pixels_semantic_grain*100.0      
        return p_of_mechanical_mask

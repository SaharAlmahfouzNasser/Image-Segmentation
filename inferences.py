from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
from build_model import *
from torch.utils.data import DataLoader
from data_loader import NucleiSegTest
from torchvision import transforms
import torch.nn.functional as F    
import numpy as np
from skimage import morphology, color, io, exposure

#weight_file = '/home/Drive2/deepak/New_whole_image_input/Final_journal_model/vggunet_he/Weights/cp_bce_interchange_custom_lr_4_mix_57_0.0443227595569.pth.tar'
weight_file = '/home/naveen/v3/Cs_Project/SaveModel/45.pth'
def IoU(y_true, y_pred):
    """Returns Intersection over Union score for ground truth and predicted masks."""
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    union = np.logical_or(y_true_f, y_pred_f).sum()
    return (intersection + 1) * 1. / (union + 1)

def Dice(y_true, y_pred):
    """Returns Dice Similarity Coefficient for ground truth and predicted masks."""
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    return (2. * intersection + 1.) / (y_true.sum() + y_pred.sum() + 1.)

def masked(img, gt, mask, alpha=1):
    """Returns image with GT lung field outlined with red, predicted lung field
    filled with blue."""
    rows, cols = img.shape[:2]
    color_mask = np.zeros((rows, cols, 3))
    boundary = morphology.dilation(gt, morphology.disk(3))^gt
    color_mask[mask == 1] = [0, 0, 1]
    color_mask[boundary == 1] = [1, 0, 0]
    
    img_hsv = color.rgb2hsv(img)
    color_mask_hsv = color.rgb2hsv(color_mask)

    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)
    return img_masked

def remove_small_regions(img, size):
    """Morphologically removes small (less than size) connected regions of 0s or 1s."""
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img

if __name__ == '__main__':

    # Path to csv-file. File should contain X-ray filenames as first column,
    # mask filenames as second column.
    # Load test data
    img_size = (1024, 1024)

    inp_shape = (1024,1024,3)
    batch_size=1

    # Load model
    #model_name = 'model.020.hdf5'
    #UNet = load_model(model_name)
    #net = UNet11(1,32,3)

    #net.load_state_dict(torch.load('/home/sahyadri/vggunet_he/Weights/cp_bce_interchange_custom_lr_4_20x_24_0.355919134286.pth.tar'))
    #net.load_state_dict(torch.load(weight_file))
    net = torch.load(weight_file)
    net.eval()
    


  
    seed = 1
    transformations_test = transforms.Compose([transforms.ToTensor()]) 
    test_set = NucleiSegTest(transforms = transformations_test)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle = False)
    ious = np.zeros(len(test_loader))
    dices = np.zeros(len(test_loader))
    folder = weight_file.split('.pth')[0].split('/')[-1]
    print (folder)
    #if not(os.path.exists('./results_custom_scratch')):
    #    os.mkdir('./results_custom_scratch')
    if not(os.path.exists('./'+folder)):
        os.mkdir('./'+folder)
        os.mkdir('./'+folder+'_th35_')
        os.mkdir('./'+folder+'_th65_')
    

    #i = 0
    for xx, yy, name in tqdm(test_loader):
        
        name = name[0][:-4]
        print (name)
        xx = xx.cuda()
        yy = yy.cuda()
        pred = net(xx)
        pred = F.sigmoid(pred)
        pred = pred.cpu()
        pred = pred.detach().numpy()[0,0,:,:]
        #mask = yy.numpy()[0,0,:,:]
        xx = xx.cpu()
        xx = xx.numpy()[0,:,:,:].transpose(1,2,0)
        img = exposure.rescale_intensity(np.squeeze(xx), out_range=(0,1))

        # Binarize masks
        #gt = mask > 0.5
        pr = pred > 0.5
        pr1 = pred >0.35
        pr2 = pred >0.65
        # Remove regions smaller than 2% of the image
        #pr = remove_small_regions(pr, 0.02 * np.prod(img_size))
       
        
        io.imsave('./'+folder + '/{}.png'.format(name), pr*255)
        io.imsave('./'+folder + '_th35_/{}.png'.format(name), pr1*255)
        io.imsave('./'+folder + '_th65_/{}.png'.format(name), pr2*255)

    #print ('Mean IoU:', ious.mean())
    #print ('Mean Dice:', dices.mean())
    
    
    

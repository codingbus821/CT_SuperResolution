import torch
import utility
import data
from trainer import Trainer
import warnings
# import vessl
import numpy as np
from tqdm import tqdm
import yaml
import os
import argparse
from model import redcnn, drunet, dncnn
from scipy.io import savemat
import measure
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='test')
parser.add_argument('--load', type=str, default='test')
parser.add_argument('--save', type=str, default='test')
args = parser.parse_args()

# vessl.configure(organization_name='2302-AI-LEC-MEDI', project_name='Minwoo-Yu')
# vessl.init(message='test_'+'redcnn_l1')

def compute_MSE(img1, img2):
    return ((img1 - img2) ** 2).mean()


def test():
    with open(os.path.join('configs', args.config+'.yaml'), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded, config_path: {}'.format(os.path.join('configs', args.config+'.yaml')))

    loader = data.Data(config, test_only=True)
    # model_test = redcnn.REDCNN().to('cuda')
    # model_test = dncnn.FDnCNN().to('cuda')
    model_test = drunet.UNetRes().to('cuda')
    model_test.load_state_dict(torch.load(os.path.join('experiment', args.load, 'model', 'model_best.pt')))
    # model_test.load_state_dict(torch.load('/home/seclab/sdd/research/joon/medical/experiment/test/REDCNN_100000iter.ckpt'))
    model_test.eval()
    data_test = loader.loader_test

    rmse_val = 0
    saver = torch.Tensor([])
    post_rmse_val = 0
    
    
    ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
    pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0

    with torch.no_grad():
        for i, (ldct, ndct) in enumerate(tqdm(data_test)):
            ldct, ndct = ldct.cuda(), ndct.cuda()
            ldct_no = utility.normalize(ldct, -500, 500)
            denoised = model_test(ldct_no)
            denoised = utility.denormalize(denoised, -500, 500)
            rmse_val += utility.calc_rmse(denoised, ndct) / len(data_test)
            saver = torch.cat([saver, denoised.cpu()], 0)
            
            
            x = ndct.cpu().detach()
            y = ldct.cpu().detach()
            pred = denoised.cpu().detach()
            
            max_dval = denoised.max()
            min_dval = denoised.min()
            
            
            ### 이미지 저장하는 함수
            normlized_denoised = (denoised - min_dval) / (max_dval - min_dval)
            save_image(normlized_denoised, f'./images/drunet/denoised/denoised_{i}.png')
            
            max_lval = ldct.max()
            min_lval = ldct.min()
            
            normlized_ldct = (ldct - min_lval) / (max_lval - min_lval)
            save_image(normlized_ldct, f'./images/drunet/ldct/ldct_{i}.png')
            
            max_nval = ndct.max()
            min_nval = ndct.min()
            
            normlized_ndct = (ndct - min_nval) / (max_nval - min_nval)
            save_image(normlized_ndct, f'./images/drunet/ndct/ndct_{i}.png')
            
            save_image(denoised, f'./denosied_test_{i}.png')
            
            
            # original_result, pred_result  = measure.compute_measure(normlized_ldct,normlized_ndct,normlized_denoised, 1)
            original_result, pred_result  = measure.compute_measure(ldct,ndct,denoised, max_nval - min_nval)
            print(i , original_result)
            print(i , pred_result)
            
            
            ori_psnr_avg += original_result[0] / len(data_test)
            ori_ssim_avg += original_result[1] / len(data_test)
            ori_rmse_avg += original_result[2] / len(data_test)
            pred_psnr_avg += pred_result[0] / len(data_test)
            pred_ssim_avg += pred_result[1] / len(data_test)
            pred_rmse_avg += pred_result[2] / len(data_test)
            print("end")
        print(ori_psnr_avg, ori_ssim_avg, ori_rmse_avg)    
        print(pred_psnr_avg, pred_ssim_avg, pred_rmse_avg)
        # numpy_arrays = [tensor.cpu().numpy() for tensor in reconstruct]
        # combined_array = np.array(numpy_arrays)
        # combined_array = np.squeeze(combined_array, axis=2)
        
        # ndct_arrays = [tensor.cpu().numpy() for tensor in reconstruct]
        # combined_ndct = np.array(ndct_arrays)
        # combined_ndct = np.squeeze(combined_ndct, axis=2)
        
        # savemat("./mat/new-unet.mat", {"recon" : combined_array})
        # savemat("./mat/ndct-mat.mat", {"ndct" : combined_ndct})
        # vessl.log(step = 0, payload={'rmse_val': rmse_val.item()})

if __name__ == '__main__':
    test()
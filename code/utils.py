import os
import json
import numpy as np
from skimage import io
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM


# test results
root_path = './figures/'
result_path = './figures/result_test/'
results = {}
final_results = {}

for folder in os.listdir(result_path):  # each folder contains an experiment result
    results.update({folder: {'psnr': [], 'ssim':[]}})
    final_results.update({folder: {'psnr_mean': 0, 'ssim_mean':0}})
    for fake_img_name in os.listdir(os.path.join(result_path, folder)):
        img_fake = io.imread(os.path.join(result_path, folder, fake_img_name), pilmode='RGB')
        psnr = []
        ssim = []
        for real_img_name in os.listdir(os.path.join(root_path, 'real_test')):
            img_real = io.imread(os.path.join(root_path, 'real_test',real_img_name), pilmode='RGB')
            psnr.append(PSNR(img_fake, img_real))
            ssim.append(SSIM(img_fake, img_real, multichannel=True))
        results[folder]['psnr'].append(np.max(psnr))  # the most similar real images
        results[folder]['ssim'].append(np.max(ssim))

    final_results[folder].update({'psnr_mean': np.mean(results[folder]['psnr'])})  # compute mean values
    final_results[folder].update({'ssim_mean': np.mean(results[folder]['ssim'])})

with open('./results/results.json', 'w') as f:
    f.write(json.dumps(results))
with open('./results/final_results.json', 'w') as f:
    f.write(json.dumps(final_results))

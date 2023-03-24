import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.utils as vutils

import numpy as np
import matplotlib.pyplot as plt

from AFGAN import NetD, NetG, initialize
from dataset import EmojiSet

# hyperparameters
EPOCHS = 500
LEARNING_RATE_G = 2e-4
LEARNING_RATE_D = 4e-4
BETA1 = 0.5  # Beta1 hyperparam for Adam optimizers (defaulted: 0.9)
BATCH_SIZE = 64  
SAMPLE_SIZE = 20
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_LENGTH = 100  # Size of z latent vector (i.e. size of generator input)
FEATURES_G = 64  # Size of feature maps in generator
FEATURES_D = 64  # Size of feature maps in discriminator
EMBEDDING_DIM = 300
REAL_LABEL = 1.
FAKE_LABEL = 0.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using {} environment...'.format(DEVICE))

# set random seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def hinge_loss(output, negtive):
    if negtive==False:
        err = torch.nn.ReLU()(1.0 - output).mean()
    else:
        err = torch.nn.ReLU()(1.0 + output).mean()
    return err


def train():
    iters = 0
    print('Start Training...')
    for epoch in range(EPOCHS):
        netG.train()
        netD.train()
        for i, batch_data in enumerate(dataloader):
            
            label_real = torch.rand(BATCH_SIZE).to(DEVICE) * 0.5 + 0.7
            # label_real = torch.full((BATCH_SIZE,), REAL_LABEL, dtype=torch.float).to(DEVICE)
            label_fake = torch.rand(BATCH_SIZE).to(DEVICE) * 0.3
            # label_fake = torch.full((BATCH_SIZE,), FAKE_LABEL, dtype=torch.float).to(DEVICE)

            
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))

            # Train with all-real batch
            netD.zero_grad()

            real_imgs = batch_data[0].to(DEVICE)
            cap_vec =  batch_data[2].to(DEVICE)

            output_real = netD(real_imgs, cap_vec).view(-1)
            loss_D_real = loss_func(output_real, label_real)  # Calculate loss on all-real batch
            # loss_D_real = loss_func(output_real, negtive=False)  # Calculate loss on all-real batch
            # loss_D_real.backward()

            # The average discriminator outputs for the all **real** batch
            # It should start close to 1 then theoretically converge to 0.5 when G gets better.
            D_x = output_real.mean().item()  

            # Train with all-fake batch
            noise = torch.randn(BATCH_SIZE, Z_LENGTH).to(DEVICE)  # Generate batch of latent vectors
            fake = netG(noise, cap_vec)   # Generate fake image batch with G

            output_fake = netD(fake.detach(), cap_vec).view(-1)
            loss_D_fake = loss_func(output_fake, label_fake)  # Calculate D's loss on the all-fake batch
            # loss_D_fake = loss_func(output_fake, negtive=True)  # Calculate D's loss on the all-fake batch
            # loss_D_fake.backward()

            # Average discriminator outputs for the all **fake** batch
            # It should start near 0 and converge to 0.5 as G gets better.
            D_G_z1 = output_fake.mean().item()  
            loss_D = (loss_D_real + loss_D_fake)  # Total loss (sum over the fake and the real batches)
            loss_D.backward()
            
            optimizerD.step()
            # schedulerD.step()


            # (2) Update G network: maximize log(D(G(z)))
            netG.zero_grad()
            
            output = netD(fake, cap_vec).view(-1)
            
            loss_G = loss_func(output, label_real)  # Calculate G's loss based on this output
            # loss_G = -output.mean()  # Calculate G's loss based on this output
            loss_G.backward()
            D_G_z2 = output.mean().item()

            optimizerG.step()
            # schedulerG.step()

            # Save Losses for plotting later
            G_losses.append(loss_G.item())
            D_losses.append(loss_D.item())

        print('[%3d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
            % (epoch+1, EPOCHS, i+1, len(dataloader), loss_D.item(), loss_G.item(), D_x, D_G_z1, D_G_z2))
        
        if (epoch % 5 == 0):
            with torch.no_grad():
                fake = netG(fixed_noise.to(DEVICE), fixed_caption_vec.to(DEVICE)).detach().cpu()
            img = vutils.make_grid(fake, nrow=4, padding=2, normalize=True)
            img_list.append(img)
            # plot_images(img, epoch)  # Reduce the frequency of cpu and gpu switching


def plot_loss():
    plt.figure(figsize=(10, 5))
    plt.title('Generator and Discriminator Loss During Training')
    plt.plot(G_losses,label='G')
    plt.plot(D_losses,label='D')
    plt.xlabel('iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('{}/loss.png'.format(figure_path))
    # plt.show()

def plot_images():
    # Plot the fake images from each epoch
    plt.figure(figsize=(4, 5), dpi=64)
    for i in range(len(img_list)):
        plt.axis('off')
        # plt.title('Fake Images from iteration {}'.format(i))
        plt.imshow(np.transpose(img_list[i],(1,2,0)))
        plt.savefig('{}/fake_{:03d}.png'.format(figure_path, i), bbox_inches='tight', pad_inches = 0.)
        # plt.show()
    plt.axis('off')
    # plt.title('Real Images')
    plt.imshow(np.transpose(vutils.make_grid(fixed_imgs, nrow=4, padding=2, normalize=True),(1,2,0)))
    plt.savefig('{}/real_imgs.png'.format(figure_path), bbox_inches='tight', pad_inches = 0.)

if __name__ == '__main__':

    # Set random seed
    setup_seed(2313516)

    # Data preperation
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(IMAGE_SIZE),
        torchvision.transforms.CenterCrop(IMAGE_SIZE),
        torchvision.transforms.ToTensor(),  # [0, 255] -> [0, 1]
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [0, 1] -> [-1, 1]
        # (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    ])

    data_path = './data/emoji_faces/'
    metadata_path = './data/metadata/idx2caption_detailed_ori.json'
    wordvector_path = './models/GoogleNews-vectors-negative300.bin.gz'
    figure_path = './figures/fake_{}'.format(data_path.split('/')[2])

    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    print('Load data set...')
    train_set = EmojiSet(os.path.join(data_path, 'train'), metadata_path, wordvector_path, transforms)
    test_set = EmojiSet(os.path.join(data_path, 'test'), metadata_path, wordvector_path, transforms)
    # dataset = EmojiSet_alpha(data_path, metadata_path, wordvector_path, 25, transforms)

    dataloader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    # Create batch of latent vectors that we will use to visualize the progression of the generator
    test_sampler = DataLoader(dataset=test_set, batch_size=SAMPLE_SIZE, shuffle=False, drop_last=True)
    fixed_noise = torch.randn(SAMPLE_SIZE, Z_LENGTH)
    fixed_imgs, fixed_caption, fixed_caption_vec = next(iter(test_sampler))
    # test_sampler = DataLoader(dataset=train_set, batch_size=80, shuffle=False, drop_last=True)  # training results
    # fixed_noise = torch.randn(80, Z_LENGTH)
    # fixed_imgs, fixed_caption, fixed_caption_vec = next(iter(test_sampler))

    # Build models
    print('Build models and initialize...')
    netG = NetG(Z_LENGTH, FEATURES_G, CHANNELS_IMG, EMBEDDING_DIM).to(DEVICE)
    netD = NetD(Z_LENGTH, FEATURES_D, CHANNELS_IMG, EMBEDDING_DIM).to(DEVICE)

    # Apply the weight initialization function to randomly initialize all weights
    netG.apply(initialize)
    netD.apply(initialize)

    # Setup Adam optimizers for both G and D
    optimizerG = torch.optim.Adam(netG.parameters(), lr=LEARNING_RATE_G, betas=(BETA1, 0.999))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=LEARNING_RATE_D, betas=(BETA1, 0.999))

    # schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=50, gamma=0.95)
    # schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=50, gamma=0.95)

    # Define loss function
    loss_func = nn.BCELoss()
    # loss_func = hinge_loss

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []

    # Calculate training time
    start =time.perf_counter()
    train()
    end = time.perf_counter()
    run_time = end - start
    hour = run_time // 3600
    minute = (run_time - 3600 * hour) // 60
    second = int(run_time - 3600 * hour - 60 * minute)
    print('Training time: {} Hours {} Minutes {} Seconds'.format(hour, minute, second))

    # save models
    torch.save(netG, './models/AFGAN_netG_{}.pth'.format(EPOCHS))
    torch.save(netD, './models/AFGAN_netD_{}.pth'.format(EPOCHS))

    # visualisation
    start =time.perf_counter()
    plot_images()
    plot_loss()
    end = time.perf_counter()
    run_time = end - start
    hour = run_time // 3600
    minute = (run_time - 3600 * hour) // 60
    second = int(run_time - 3600 * hour - 60 * minute)
    print('Plotting time: {} Hours {} Minutes {} Seconds'.format(hour, minute, second))

    


import numpy as np
import cv2
import torch

def transform_image_pair(input_data, input_size, maxAng, angDiff):
    c_dim = input_data.shape[1]
    imgOut_0 = torch.zeros((c_dim, input_size, input_size))
    imgOut_1 = torch.zeros((c_dim, input_size, input_size))
    imgTemp = np.pad(input_data[0], ((2, 2), (2, 2)), 'constant', constant_values=((0, 0), (0, 0)))

    angle_use_0 = int(np.random.randint(low = 0,high = maxAng, size = 1))
    angle_use_1 = angle_use_0 + angDiff
    img_size = imgTemp.shape[0]

    M_0 = cv2.getRotationMatrix2D((img_size/2, img_size/2), angle_use_0, 1)
    imgOut_0 = torch.tensor(cv2.warpAffine(imgTemp, M_0, (img_size, img_size)))[2:-2, 2:-2].unsqueeze(0)
    M_1 = cv2.getRotationMatrix2D((img_size/2, img_size/2), angle_use_1, 1)
    imgOut_1 = torch.tensor(cv2.warpAffine(imgTemp, M_1, (img_size, img_size)))[2:-2, 2:-2].unsqueeze(0)

    return imgOut_0, imgOut_1

def compute_cluster_centers(train_loader, latent_dim, num_labels, encoder, device):
    cluster_center_sum = torch.zeros(num_labels, latent_dim).to(device)
    num_class_image = torch.zeros(num_labels).to(device)
    with torch.no_grad():
        for idx, batch in enumerate(train_loader):
            image, _, label = batch
            image = image.to(device)

            z = encoder(image)

        for k in range(num_labels):
            class_idx = np.where(label == k)[0]
            latent_vec = z[class_idx,:]
            cluster_center_sum[k, :] = cluster_center_sum[k, :] + latent_vec.sum(dim=0)
            num_class_image[k] = num_class_image[k] + class_idx.shape[0]

    return cluster_center_sum / num_class_image[:, None]



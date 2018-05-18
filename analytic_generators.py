import numpy as np
import scipy as scipy

#This package simulates quantum and electronic noise kernels as described in Wang AS, Stayman JW, Otake Y, Vogt S, Kleinszig G, Khanna AJ, et al. Low-dose preview for patient-specific, task-specific technique selection in cone-beam CT. Med Phys

kernel_shot_noise = np.array([[0.03, 0.06, 0.02], [0.11, 0.98, 0.11], [0.02, 0.06, 0.03]])#A. S. Wang et al., “Low-dose preview for patient-specific, task-specific technique selection in cone-beam CT,” Med Phys, vol. 41, no. 7, Jul. 2014.
factor_electronic_noise = 0
kernel_electronic_noise = np.array([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.01, 0.01, 0.98, 0.01, 0.01], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])#A. S. Wang et al., “Low-dose preview for patient-specific, task-specific technique selection in cone-beam CT,” Med Phys, vol. 41, no. 7, Jul. 2014.

def add_noise(input_image, photon_prob, photon_number):
    output_image = np.zeros(input_image.shape)
    for i in range(0, input_image.shape[0]):
        shot_noise = (np.random.poisson(photon_prob[i, :, :] * photon_number) - photon_prob[i, :, :] * photon_number) * input_image[i, :, :] / (photon_prob[i, :, :] * photon_number)
        shot_noise = scipy.signal.convolve2d(shot_noise, kernel_shot_noise, mode="same")
        electronic_noise = np.random.randn(input_image.shape[1], input_image.shape[2]) * factor_electronic_noise
        electronic_noise = scipy.signal.convolve2d(electronic_noise, kernel_electronic_noise, mode="same")
        output_image[i,:,:] = np.clip(input_image[i, :, :] + shot_noise + electronic_noise, a_min=0.0, a_max = 10e30)
    return output_image
import numpy as np
import PIL.Image as Image
from datetime import datetime
from os import path
import pickle


def image_saver(images, prefix, path):
    for i in range(0, images.shape[0]):
        image_pil = Image.fromarray(images[i, :, :])
        image_pil.save(path + "\\" + prefix + str(i).zfill(5) + ".tiff")
    return True


def param_saver(thetas, phis, proj_mats, camera, origin, photons, spectrum, prefix, save_path):
    i0 = np.sum(spectrum[:, 0] * (spectrum[:, 1] / np.sum(spectrum[:, 1]))) / 1000
    data = {"date": datetime.now(), "thetas": thetas, "phis": phis, "proj_mats": proj_mats, "camera": camera, "origin": origin, "photons": photons, "spectrum": spectrum, "I0": i0}
    with open(path.join(save_path, prefix + '.pickle'), 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    return True



class Camera():
    def __init__(self, sensor_width, sensor_height, pixel_size, source_to_detector_distance, isocenter_distance):
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        self.pixel_size = pixel_size
        self.source_to_detector_distance = source_to_detector_distance
        self.isocenter_distance = isocenter_distance
        raise DeprecationWarning("Use deepdrr.geometry.Camera instead")        

    def __str__(self):
        return f"""\
Camera(sensor_width = {self.sensor_width},
       sensor_height = {self.sensor_height},
       pixel_size = {self.pixel_size},
       source_to_detector_distance = {self.source_to_detector_distance},
       isocenter_distance = {self.isocenter_distance})"""

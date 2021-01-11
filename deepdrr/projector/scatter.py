import logging
import torch
import numpy as np
from skimage.transform import resize
import torchvision.transforms as transforms
from pathlib import Path

from .network_scatter import SimpleNetGenerator


logger = logging.getLogger(__name__)


class ScatterNet():
    def __init__(self):
        torch.cuda.set_device(0)
        d = Path(__file__).resolve().parent
        self.model_path = d / "model_scatter.pth"
        self.model = SimpleNetGenerator()
        self.model.load_state_dict(torch.load(self.model_path))
        self.model = self.model.cuda()
        self.transform = transforms.Compose([transforms.ToTensor()])
        logger.info("loaded scatter net.")

    def add_scatter(self, input_image, camera):
        scale_factor = 0.1334 * camera.pixel_size[0] / 0.31
        # input_image = np.ascontiguousarray(np.swapaxes(input_image,0,1))
        input_image = np.ascontiguousarray(np.swapaxes(input_image, 1, 2))
        maxlen = np.max(input_image.shape[1:-1]) * scale_factor
        uselen = np.int(np.power(2, np.ceil(np.log2(maxlen))))
        max = np.max(input_image)
        tensor_list = []
        cuts = []
        for i in range(0, input_image.shape[0]):
            image = resize(input_image[i, :, :] / max, [np.int(np.round(input_image.shape[1] * scale_factor)), np.int(np.round(input_image.shape[2] * scale_factor))]) * max * 256 / 5.5
            cuts = (((uselen - image.shape[0]) // 2, (uselen - image.shape[0]) - (uselen - image.shape[0]) // 2), ((uselen - image.shape[1]) // 2, (uselen - image.shape[1]) - (uselen - image.shape[1]) // 2))
            image = np.pad(image, cuts, mode='reflect')
            image.shape = (image.shape[0], image.shape[1], 1)
            tensor_list.append(self.transform(image).unsqueeze(0))
        image_tensor = torch.cat(tensor_list).cuda()
        output = self.model.forward(image_tensor) * 0.10
        scatter = output.cpu().detach().numpy()
        scatter = np.squeeze(scatter, 1)
        # cut relevant part
        scatter = scatter[:, cuts[0][0]:uselen - cuts[0][1], cuts[1][0]:uselen - cuts[1][1]]
        out = np.zeros(input_image.shape)
        for i in range(0, scatter.shape[0]):
            # edge compensation models decreasing scatter intensity at the borders of the detector (only working for 1240*960 0.31mm detector)
            # comp_scatter = edge_compensation(scatter[i, :, :])
            # out[i,:,:]=resize(comp_scatter, input_image[i, :, :].shape, order=3, mode="edge")
            out[i, :, :] = resize(scatter[i, :, :], input_image[i, :, :].shape, order=3, mode="edge")
        out = np.swapaxes(out, 1, 2)
        return out


def edge_compensation(img):
    edge = np.sin(np.linspace(0, np.pi / 2, 45, dtype=np.float32))
    window_h = np.ones((128), dtype=np.float32)
    window_h[0:45] = edge
    window_h[128 - 45:128] = edge[::-1]
    window_h = window_h * 0.35 + 0.65
    window_w = np.ones((165), dtype=np.float32)
    window_w[0:45] = edge
    window_w[165 - 45:165] = edge[::-1]
    window_w = window_w * 0.35 + 0.65
    window = np.outer(window_w, window_h)
    return img * window


if __name__ == "__main__":
    scatter = ScatterNet()


from contextlib import contextmanager

import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
import deepdrr
from deepdrr import geo
from deepdrr.utils import test_utils
from PIL import Image
import pytest
import copy
import time
import matplotlib.pyplot as plt
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageSequence
from pathlib import Path
import io
from matplotlib import pyplot as plt

import pyvista as pv
import logging
import pyrender
from deepdrr.pyrenderdrr.material import DRRMaterial
from deepdrr.utils.mesh_utils import *

@contextmanager
def verify_image(name, actual_dir, expected_dir, diff_dir, atol=1):
    actual_dir = Path(actual_dir)
    expected_dir = Path(expected_dir)
    diff_dir = Path(diff_dir)

    if (actual_dir / name).exists():
        (actual_dir / name).unlink()

    yield actual_dir / name

    diff_dir.mkdir(parents=True, exist_ok=True)

    try:
        actual_img = Image.open(actual_dir / name)
    except FileNotFoundError:
        print(f"Actual image not found: {actual_dir / name}")
        pytest.fail("Actual image not found")

    try: 
        expected_img = Image.open(expected_dir / name)
    except FileNotFoundError:
        print(f"Truth image not found: {expected_dir / name}")
        pytest.skip("Truth image not found")

    actual_frames = []
    for actual_frame in ImageSequence.Iterator(actual_img):
        # actual_frames.append(np.array(actual_frame)) # ValueError: No packer found from P to L
        actual_frames.append(np.array(actual_frame.convert('RGB')))

    expected_frames = []
    for expected_frame in ImageSequence.Iterator(expected_img):
        # expected_frames.append(np.array(expected_frame))
        expected_frames.append(np.array(expected_frame.convert('RGB')))

    assert len(actual_frames) == len(expected_frames), f"Number of frames in actual and expected images do not match: {len(actual_frames)} != {len(expected_frames)}"
    
    diff_ims = []
    max_diff = 0
    min_diff = 0
    for i, expected_frame in enumerate(expected_frames):
        actual_frame = actual_frames[i]
        if not np.allclose(actual_frame, actual_frame[:, :, 0][:, :, np.newaxis]):
            print(f"Warning: Image {i} has different values in different channels, this compare function only shows the first channel diff")
        diff_im = actual_frame.astype(np.float32) - expected_frame.astype(np.float32)
        max_diff = max(max_diff, diff_im.max())
        min_diff = min(min_diff, diff_im.min())
        diff_ims.append(diff_im)

    same = True
    for i, diff_im in enumerate(diff_ims):
        if not np.allclose(diff_im, 0, atol=atol):
            same = False
            break

    diff_name = Path(name).stem+"_diff"
    diff_path = diff_dir / (diff_name + ".png")
    if not same:  
        pil_fig_imgs = []
        for i, diff_im in enumerate(tqdm.tqdm(diff_ims)):
            plt.figure()
            plt.imshow(diff_im[:,:,0], cmap="viridis", vmin=min_diff, vmax=max_diff)
            plt.colorbar()
            # diff_name = f"diff_{name}"
            # if len(expected_frames) > 1:
            #     diff_name = f"{diff_name}_diff_{i:03d}"
    
            # plt.savefig(self.output_dir / (diff_name + ".png"))
            # save figure to PIL Image
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            
            buf.seek(0)
            pil_fig_img = Image.open(buf)
            pil_fig_imgs.append(pil_fig_img)
        
        if len(pil_fig_imgs) > 1:
            pil_fig_imgs[0].save(
                diff_path,
                save_all=True,
                append_images=pil_fig_imgs[1:],
                duration=expected_img.info.get('duration', 100),
                loop=0,  # 0 means loop indefinitely, you can set another value if needed
                disposal=1,  # 2 means replace with background color (use 1 for no disposal)
            )
        else:
            pil_fig_imgs[0].save(diff_path)

    else:
        # write a green image
        passed_img = Image.new('RGB', (expected_img.width, expected_img.height), color = (0, 255, 0))
        passed_img.save(diff_path)

    # for i, diff_im in enumerate(diff_ims):
    #     assert np.allclose(diff_im, 0, atol=atol), f"Test {name} failed"
    if not same:
        pytest.fail(f"Images do not match: {name}")

    print(f"Test {name} passed")
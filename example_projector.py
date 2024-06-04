#! python3
"""Minimal projection example with DeepDRR."""

import deepdrr
from deepdrr import geo
from deepdrr.utils import test_utils, image_utils
from deepdrr.projector import Projector
from deepdrr.vol import Mesh
from deepdrr.pyrenderdrr import DRRMaterial


def main():
    output_dir = test_utils.get_output_dir()
    data_dir = test_utils.download_sampledata("CTPelvic1K_sample")
    ct = deepdrr.Volume.from_nifti(
        data_dir / "dataset6_CLINIC_0001_data.nii.gz", use_thresholding=True
    )
    ct.supine()

    # define the simulated C-arm
    carm = deepdrr.device.SimpleDevice()

    tool_path = "data/6.5mmD_32mmThread_L130mm.STL"
    mesh: Mesh = Mesh.from_stl(tool_path)
    dense_mesh = Mesh.from_stl(tool_path, material=DRRMaterial("iron", density=7.87))

    # project in the anterior direction
    with Projector(
        [ct, mesh, dense_mesh], device=carm, intensity_upper_bound=4
    ) as projector:
        p = ct.center_in_world
        v = ct.world_from_anatomical @ geo.vector(0, 1, 0)
        carm.set_view(
            p,
            v,
            up=ct.world_from_anatomical @ geo.vector(0, 0, 1),
            source_to_point_fraction=0.7,
        )

        mesh.place_center(ct.center_in_world)

        image = projector()

    path = output_dir / "example_projector.png"
    image_utils.save(path, image)
    print(f"saved example projection image to {path.absolute()}")


if __name__ == "__main__":
    main()

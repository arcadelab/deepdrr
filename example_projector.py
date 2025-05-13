#! python3
"""Minimal projection example with DeepDRR."""

import deepdrr
from deepdrr.utils import test_utils, image_utils
from deepdrr.projector import Projector
from deepdrr.vol import Mesh
from deepdrr.pyrenderdrr import DRRMaterial
import killeengeo as kg


def main():
    output_dir = test_utils.get_output_dir()
    print("loading ct")
    ct = deepdrr.Volume.from_nifti(
        "/mnt/oracle_data/killeen/NMDID-ARCADE_2024-12-08/nifti/case-100065/LOWER_EXTREMITY/THIN_BONE_L-EXT_LOWER_EXTREMITY_Orthoped_73918_11764.nii.gz",
        # "/home/killeen/Downloads/case-100366/real-case-100366/nifti/FULL_BODY_COMBINED/case-100366/case-100366.nii.gz",
        use_thresholding=True,
    )
    ct.supine()

    # define the simulated C-arm
    device = deepdrr.device.SimpleDevice()

    tool_path = "data/6.5mmD_32mmThread_L130mm.STL"
    mesh: Mesh = Mesh.from_stl(tool_path)
    dense_mesh = Mesh.from_stl(tool_path, material=DRRMaterial("iron", density=7.87))

    print("initializing projector")
    # project in the anterior direction
    with Projector([ct], device=device, intensity_upper_bound=4) as projector:
        # p = ct.center_in_world
        # v = ct.world_from_anatomical @ geo.vector(0, 1, 0)
        p = kg.p(0, 0, 0)
        v = kg.v(0, 0, 1)
        device.set_view(
            p,
            v,
            source_to_point_fraction=0.7,
        )

        mesh.place_center(p)

        image = projector()

    path = output_dir / "example_projector.png"
    image_utils.save(path, image)
    print(f"saved example projection image to {path.absolute()}")


if __name__ == "__main__":
    main()

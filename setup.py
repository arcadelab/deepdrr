import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepdrr",
    version="1.1.0a3",
    author="Benjamin D. Killeen",
    author_email="killeen@jhu.edu",
    description="A Catalyst for Machine Learning in Fluoroscopy-guided Procedures.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=[
        "colorlog",
        "numpy",
        "scipy",
        "scikit-image",
        "torch",
        "torchvision",
        "nibabel",
        "pydicom",
        "pynrrd",
        "pyvista",
        "pycuda",
        "rich",
    ],
    include_package_data=True,
    python_requires=">=3.7",
)

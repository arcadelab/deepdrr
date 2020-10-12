import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='deepdrr',
    version='0.1.dev0',
    author='Mathias Unberath',
    author_email='unberath@jhu.edu',
    description='Forked from mathiasunberath/DeepDRR to be more pythonic.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires='>=3.8',
)

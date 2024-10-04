import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nely-pyba",
    version="0.13.1",
    author="Semih Günel",
    packages=["pyba"],
    description="Python Bundle Adjustment Routines",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NeLy-EPFL/PyBundleAdjustment",
    install_requires=[
        "opencv-python-headless>=4.8.1.78", # https://github.com/NeLy-EPFL/DeepFly3D/security/dependabot/4
        "numpy",
        "matplotlib",
        "scipy"
    ],
)

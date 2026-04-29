from setuptools import find_packages, setup


setup(
    name="pg-sam-inference",
    version="1.0.0",
    description="Self-contained PG-SAM / MMA-SAM2 inference package.",
    packages=find_packages(include=["map_sam2", "map_sam2.*", "sam2", "sam2.*"]),
    include_package_data=True,
    package_data={"sam2": ["configs/**/*.yaml"]},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.5.1",
        "numpy>=1.24.4",
        "pillow>=9.4.0",
        "hydra-core>=1.3.2",
        "iopath>=0.1.10",
        "omegaconf>=2.3.0",
        "scipy>=1.10.0",
        "opencv-python-headless>=4.7.0",
    ],
)

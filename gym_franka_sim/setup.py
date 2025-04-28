from setuptools import setup, find_packages

setup(
    name="gym_franka_sim",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.28.1",
        "numpy>=1.20.0",
        "torch>=1.13.0",
        "mujoco>=2.3.0",
    ],
    python_requires=">=3.8",
    zip_safe=False,
)

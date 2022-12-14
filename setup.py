from setuptools import find_packages, setup

package_name = "mup_demo"
packages = find_packages(where=package_name)

# Stuff used to make this work as a pip-installable gist:
# if __name__ == "__main__":
#     # The name of the file next to this (without the .py)
#     if not os.path.exists(file_name):
#         os.mkdir(file_name)
#     shutil.copyfile(f"{file_name}.py", f"{file_name}/__init__.py")

setup(
    name=package_name,
    version="0.0.1",
    description="Simple training script for testing multi-GPU stuff on the Mila cluster.",
    author="Fabrice Normandin",
    author_email="normandf@mila.quebec",
    packages=[package_name],  # Same as name
    python_requires=">=3.9",
    # entry_points={
    #     "console_scripts": [
    #         f"{file_name} = {file_name}:main",
    #     ]
    # },
    # External packages as dependencies
    install_requires=[
        "mup @ git+https://github.com/microsoft/mup.git",
        "mutransformers @ git+https://github.com/microsoft/mutransformers.git",
        "transformers",
        "evaluate",
        "scikit-learn",
        "datasets",
        "accelerate",
        "torchvision",
        "pytorch-lightning==1.6.0",
        "lightning-bolts==0.5",
        "simple-parsing",
        "filelock",
        "orion",
        "wandb",
    ],
    extras_require={
        "fairscale": ["fairscale"],
    },
)

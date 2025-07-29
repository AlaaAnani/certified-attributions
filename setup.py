from setuptools import setup, find_packages

setup(name='attribution_certification',
      version='1.0',
      packages=find_packages(),
      install_requires=["captum == 0.4.1",
                        "matplotlib == 3.5.1",
                        "numpy == 1.21.2",
                        "pandas == 1.3.5",
                        "scikit_image == 0.19.1",
                        "seaborn == 0.11.2",
                        "torch == 1.13.1",
                        "torchvision == 0.14.1",
                        "tqdm == 4.62.3",
                        "pickle5 == 0.0.12",
                        "zennit == 0.4.7",
                        "jupyter == 1.1.1",
                        "two-sample-binomial == 0.0.4",
                        "opencv-python == 4.12.0.88"])

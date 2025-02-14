from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="hf-downloader",
    version="0.1.0",
    author="Mubarak H. Alketbi",
    author_email="mubarak.harran@gmail.com",
    description="Advanced Hugging Face model/downloader with smart resource management",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MubarakHAlketbi/hf_downloader",
    packages=find_packages(),
    install_requires=[
        "requests>=2.26.0",
        "huggingface_hub>=0.11.0",
        "tqdm>=4.62.0",
        "portalocker>=2.3.2",
        "blake3>=0.3.0"
    ],
    entry_points={
        "console_scripts": [
            "hf_downloader=hf_downloader.cli:main"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
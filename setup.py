from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="hf-downloader",
    version="0.1.0",
    author="Your Name",
    author_email="your@email.com",
    description="Advanced Hugging Face model/downloader with smart resource management",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hf-downloader",
    packages=find_packages(),
    install_requires=[
        "requests",
        "huggingface_hub",
        "tqdm",
        "portalocker",
        "blake3"
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
    python_requires='>=3.7',
)

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hfdl",
    version="0.1.21",
    author="Mubarak H. Alketbi",
    author_email="mubarak.harran@gmail.com",
    description="Advanced Hugging Face model/downloader with smart resource management",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MubarakHAlketbi/hfdl",
    packages=find_packages(),
    install_requires=[
        "requests>=2.32.3",
        "huggingface_hub >=0.28.1",
        "tqdm>=4.67.1",
        "portalocker>=3.1.1",
        "blake3>=1.0.4",
        "pydantic>=2.10.6",
        "typing-extensions>=4.12.2"
    ],
    entry_points={
        "console_scripts": [
            "hfdl=hfdl.cli:main"
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires='>=3.10',
)
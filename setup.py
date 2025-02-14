from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hfdl",
    version="0.1.9",
    author="Mubarak H. Alketbi",
    author_email="mubarak.harran@gmail.com",
    description="Advanced Hugging Face model/downloader with smart resource management",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MubarakHAlketbi/hfdl",
    packages=find_packages(),
    install_requires=[
        "requests>=2.26.0",
        "huggingface_hub>=0.11.0",
        "tqdm>=4.62.0",
        "portalocker>=2.3.2",
        "blake3>=0.3.0",
        "pydantic>=2.0.0",
        "typing-extensions>=4.0.0"
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
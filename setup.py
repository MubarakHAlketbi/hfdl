from setuptools import setup, find_packages

# Define requirements directly
core_requirements = [
    "huggingface_hub>=0.28.1",
    "tqdm>=4.67.1",
    "pydantic>=2.10.6",
    "requests>=2.32.3"
]

test_requirements = [
    "pytest>=8.3.4",
    "pytest-mock>=3.14.0",
    "types-requests>=2.32.0",
    "types-tqdm>=4.67.0"
]

dev_requirements = [
    "black>=25.1.0",
    "isort>=6.0.0",
    "mypy>=1.15.0",
    "flake8>=7.1.1"
]

setup(
    name="hfdl",
    version="0.3.1",
    description="Fast and reliable downloader for Hugging Face models and datasets",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author="Mubarak H. Alketbi",
    author_email="mubarak.harran@gmail.com",
    url="https://github.com/MubarakHAlketbi/hfdl",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'hfdl=hfdl.cli:main',
        ],
    },
    install_requires=core_requirements,
    extras_require={
        'test': test_requirements,
        'dev': test_requirements + dev_requirements,
    },
    python_requires='>=3.10',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='huggingface download models datasets machine-learning',
    project_urls={
        'Bug Reports': 'https://github.com/MubarakHAlketbi/hfdl/issues',
        'Source': 'https://github.com/MubarakHAlketbi/hfdl',
    },
    include_package_data=True,
    zip_safe=False,
)
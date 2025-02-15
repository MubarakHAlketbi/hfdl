from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Split requirements into core and test
core_requirements = [req for req in requirements if not any(
    x in req for x in ['pytest', 'black', 'isort', 'mypy', 'flake8', 'types-']
)]
test_requirements = [req for req in requirements if any(
    x in req for x in ['pytest', 'types-']
)]
dev_requirements = [req for req in requirements if any(
    x in req for x in ['black', 'isort', 'mypy', 'flake8']
)]

setup(
    name="hfdl",
    version="0.3.0",
    description="Fast and reliable downloader for Hugging Face models and datasets",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/hfdl",
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
        'Bug Reports': 'https://github.com/yourusername/hfdl/issues',
        'Source': 'https://github.com/yourusername/hfdl',
    },
    include_package_data=True,
    zip_safe=False,
)
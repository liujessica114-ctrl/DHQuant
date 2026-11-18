"""Package the DHQuant research code and its vendored BasicSR modules."""
from pathlib import Path
from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent
requirements = [line.strip() for line in (ROOT / 'requirements.txt').read_text().splitlines()
                if line.strip() and not line.startswith('#')]
setup(
    name='dhquant',
    version=(ROOT / 'VERSION').read_text().strip(),
    description='Smoothing-Guided Channel Grouping Quantization for Super-Resolution Transformers',
    long_description=(ROOT / 'README.md').read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    packages=find_packages(include=['basicsr', 'basicsr.*']),
    python_requires='>=3.8',
    install_requires=requirements,
    license='Apache-2.0',
    license_files=['LICENSE', 'NOTICE'],
)

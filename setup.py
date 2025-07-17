from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="comic-panel-extractor",
    version="0.1.0",
    author="Jebin Einstein E",
    author_email="jebineinstein@gmail.com",
    description="A tool for extracting panels from comic book images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jebin2/comic-panel-extractor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "comic-panel-extractor=comic_panel_extractor.cli:main",
            "serve-comic-panel-extractor=comic_panel_extractor.server:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
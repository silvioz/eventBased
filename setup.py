import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="eventBased",
    version="0.0.1",
    author="Silvio Zanoli",
    author_email="Silvio.zanoli@epfl.ch",
    description="A packages to menage event-based signals",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://c4science.ch/diffusion/9595/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL License",
    ],
    python_requires='>=3.6',
)

import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

print(find_packages())

# This call to setup() does all the work
setup(
    name="bacpack",
    version="0.0.17",
    description="Beta Adjusted Covariance estimation",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Kirill Dragun, Kris Boudt",
    author_email="kdragun@vub.be, kris.boudt@vub.be",
    packages=["bacpack"],
    include_package_data=True,
    keywords='BAC estimator highfrequency high frequency bacpack',
    install_requires=["numpy", "pandas", "numba"]
)

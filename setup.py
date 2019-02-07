from setuptools import setup, find_packages
import pip

setup(
    name='py_object_detection',
    version='0.0.1',
    description="Common data interface",
    url='https://github.com/uniquetrij/py-tensorflow-runner',
    author='Trijeet Modak',
    author_email='uniquetrij@gmail.com',
    packages=find_packages(),
    include_package_data = True,
    zip_safe=False
)

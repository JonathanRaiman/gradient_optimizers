import os
from setuptools import setup, find_packages

def readfile(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='gradient-optimizers',
    version='0.0.3',
    description='Python package for wrapping gradient optimizers for models in Theano',
    long_description=readfile('README.md'),
    ext_modules=[],
    packages=find_packages(),
    py_modules = [],
    author='Jonathan Raiman',
    author_email='jraiman at mit dot edu',
    url='https://github.com/JonathanRaiman/gradient_optimizers',
    download_url='https://github.com/JonathanRaiman/gradient_optimizers',
    keywords='Machine Learning, Gradient Descent, NLP, Optimization, Hessian Free optimization',
    license='MIT',
    platforms='any',
    zip_safe=False,
    classifiers=[
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.3',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    # test_suite="something.test",
    setup_requires = [],
    install_requires=[
        'theano',
        'numpy'
    ],
    include_package_data=True,
)
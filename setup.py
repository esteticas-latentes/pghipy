from setuptools import setup

def readme():
    with open('README.md') as fh:
        return fh.read()

setup(
    name='pghipy',
    version='0.1.1',
    url='https://github.com/esteticas-latentes/pghipy',
    author='Laurence Bender, Leonardo Pepino',
    author_email='pghipy@gmail.com',
    description='STFT/ISTFT transforms and phase recovery using Phase Gradient Heap Integration',
    long_description=readme(),
    long_description_content_type="text/markdown",
    license='MIT',
    packages=['pghipy'],    
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Microsoft :: Windows :: Windows 10',   
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering'
    ],
    keywords=['pghi', 'stft', 'istft', 'spectrogram', 'phase recovery'],
    install_requires=['numpy', 'scipy', 'numba'],
    python_requires='>=3.6',
    zip_safe=False
)

from setuptools import setup, find_packages

setup(
    name='OptionGreeksGPU',
    version='0.1.0',
    author='Salil Bhasin',
    author_email='salilbhasinofficial@gmail.com',
    description='GPU-accelerated computation of option Greeks',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/SalilBhasinOfficial/OptionGreeksGPU.git',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'cupy',
        'numba',
        'pandas',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Financial and Insurance Industry',
        'Topic :: Office/Business :: Financial :: Investment',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.7',
    keywords='options greeks computation gpu acceleration finance options trading quantitative finance risk management financial engineering option pricing CUDA GPU computing high performance computing derivatives trading volatility hedging strategies investment analysis trading algorithms financial markets black-scholes algorithmic trading portfolio optimization market analysis'
)

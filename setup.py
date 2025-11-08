from setuptools import setup, find_packages

setup(
    name='OptionGreeksGPU',
    version='3.0.0',
    author='Salil Bhasin',
    author_email='salilbhasinofficial@gmail.com',
    description='High-performance option Greeks computation with JAX automatic differentiation and GPU/TPU acceleration',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/SalilBhasinOfficial/OptionGreeksGPU.git',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20.0',
        'numba>=0.55.0',
        'pandas>=1.3.0',
    ],
    extras_require={
        'jax': [
            'jax>=0.4.0',
            'jaxlib>=0.4.0',
        ],
        'jax-cuda': [
            'jax[cuda12]>=0.4.0',
        ],
        'jax-tpu': [
            'jax[tpu]>=0.4.0',
        ],
        'all': [
            'jax>=0.4.0',
            'jaxlib>=0.4.0',
            'scipy>=1.7.0',
        ],
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Financial and Insurance Industry',
        'Topic :: Office/Business :: Financial :: Investment',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    python_requires='>=3.8',
    keywords='options greeks computation gpu acceleration finance jax automatic differentiation options trading quantitative finance risk management financial engineering option pricing CUDA GPU computing high performance computing derivatives trading volatility hedging strategies investment analysis trading algorithms financial markets black-scholes algorithmic trading portfolio optimization market analysis tpu xla'
)

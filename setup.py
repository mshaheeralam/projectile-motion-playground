from setuptools import setup, find_packages

setup(
    name='projectile-motion-playground',
    version='0.1.0',
    author='Shaheer Alam',
    author_email='mshaheeralamkz@gmail.com',
    description='A playground for simulating and predicting projectile motion with air drag using machine learning.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'matplotlib',
        'torch',
        'streamlit',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
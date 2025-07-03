from setuptools import setup, find_packages

setup(
    name='sensor_fault_detection',
    version='0.1.0',
    author='Nitin Savio Bada',
    author_email='nitinsaviobada@gmail.com',
    description='A Scania Truck Sensor Fault Detection System using Machine Learning and Flask',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/NITIN9181/sensor-fault-detection-scania',
    project_urls={
        "Bug Tracker": "https://github.com/NITIN9181/sensor-fault-detection-scania/issues",
    },
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "flask",
        "matplotlib",
        "joblib",
        "gunicorn",  
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", 
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)

from setuptools import setup

with open('README.md', 'r') as f:
    readme = f.read()



setup(
    name="cleanX",
    version='0.0.2',
    description="Python library for cleaning data in large datasets of Xrays",
    long_description=readme,
    long_description_content_type='text/markdown',
    author='doctormakeda@gmail.com',
    author_email='doctormakeda@gmail.com',
    maintainer='doctormakeda@gmail.com',
    maintainer_email= 'doctormakeda@gmail.com',
    url="https://github.com/drcandacemakedamoore/cleanX",
    license="MIT",
    py_modules=["cleanX"],
    
    install_requires=[
        "pandas",
        'numpy',# < 3.0 ; python_version < "3.6"',
        "matplotlib",
        "pillow",
        "tesserocr",
        
        
    ],
   
    
    



)

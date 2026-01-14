from setuptools import setup, find_packages

setup(
    name="ecommerce-intelligence-platform",
    version="0.1.0",
    author="Priyanshu Patra",
    author_email="priyanshupatra22072002@gmail.com",
    description="Enterprise E-Commerce Intelligence & Recommendation Platform",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/01Priyanshu/ecommerce-intelligence-platform",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
)

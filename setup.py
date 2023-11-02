from setuptools import setup, find_namespace_packages

setup(name="pvmcuda_pkg",
      version="0.0.1",
      description="GPU implementation of Predictive Vision Model",
      author="Filip Piekniewski",
      packages=find_namespace_packages(),
      include_package_data=True,
      entry_points={
          "console_scripts": [
              "pvm=pvmcuda_pkg.run:execute",
          ],
      },
      zip_safe=False
      )


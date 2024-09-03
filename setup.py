from os import path, walk
from setuptools import setup, find_packages


package_name = "mj_pin_wrapper"
package_version = "1.0.0"

def find_resources(package_name):
    """Find the relative path of files under the resource folder."""
    resources = []
    package_dir = path.join("src", package_name)
    resources_dir = path.join(package_dir, "resources")

    for (root, _, files) in walk(resources_dir):
        for afile in files:
            if (
                afile != package_name
                and not afile.endswith(".DS_Store")
                and not afile.endswith(".py")
            ):
                rel_dir = path.relpath(root, package_dir)
                src = path.join(rel_dir, afile)
                resources.append(src)
    return resources


with open(path.join(path.dirname(path.realpath(__file__)), "README.md"), "r") as fh:
    long_description = fh.read()

# Find the resource files.
resources = find_resources(package_name)

# Install nodes and demos.
scripts_list = []
for (root, _, files) in walk(path.join("demos")):
    for demo_file in files:
        scripts_list.append(path.join(root, demo_file))

# Setup the package
setup(
    name=package_name,
    version="1.0.0",
    package_dir={
        "": "src",
    },
    packages=find_packages(where="src"),
    package_data={package_name: resources},
    scripts=scripts_list,
    install_requires=["mujoco", "robot_descriptions"],
    zip_safe=True,
    maintainer="",
    maintainer_email="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    description="",
)
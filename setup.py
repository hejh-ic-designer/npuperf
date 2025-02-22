# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, exec-used
"""Setup HHB package."""

from setuptools import find_packages
from setuptools import setup
from npuperf import __version__

def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content

setup(
    name="npuperf",
    version=__version__,
    description="npuperf: a DLA perf tool.",
    long_description=readme(),
    long_description_content_type='text/markdown',
    zip_safe=False,
    install_requires=[
        "numpy",
        "networkx",
        "sympy",
        "matplotlib",
        "multiprocessing_on_dill",
    ],
    python_requires='>=3.10',
    packages=find_packages(),
    include_package_data=True,
)

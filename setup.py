# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Install script for setuptools."""

from setuptools import find_packages
from setuptools import setup

setup(
    name='tell_me_why_explanations_rl',
    version='1.0',
    description='TODO(lampinen)',
    author='DeepMind',
    author_email='laminen@deepmind.com',
    license='Apache License, Version 2.0',
    url='https://github.com/deepmind/tell_me_why_explanations_rl',
    packages=find_packages(),
    install_requires=[
        'absl-py',
        'dm-env',
        'numpy',
        'pycolab'
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)

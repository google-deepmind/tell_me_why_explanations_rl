# Tell me why! Some environments for explanations in RL

This repository contains the source code for some environments used in our paper
"Tell me why! Explanations support learning relational and causal structure"
(https://arxiv.org/abs/2112.03753). In particular, this contains the
implementation for the 2D odd-one-out environment with the basic, confounded,
and experimenting/meta-learning versions.

## Installation

You can install this package directly from GitHub.
We recommend installing in a fresh Python virtual environment:

```shell
python3 -m venv tell_me_why
source tell_me_why/bin/activate
pip install --upgrade pip
pip install git+git://github.com/deepmind/tell_me_why_explanations_rl.git.
```

## Usage

To import the 2D environment and load the basic training levels with full
explanations you can run the following code (from `examples/example.py'):

```python
from tell_me_why_explanations_rl import odd_one_out_environment

train_levels = ['color_full', 'shape_full', 'texture_full', 'position_full']
train_envs = [odd_one_out_environment.from_name(l) for l in train_levels]

# try one out
env = train_envs[0]
timestep = env.reset()
for action in range(8):
  timestep = env.step(action)
```

To train without explanations, use:

```python
train_levels = ['color_none', 'shape_none', 'texture_none', 'position_none']
```


For the confounding experiments, use the following train/test levels:

```python
train_levels = ['confounding_color_full', 'confounding_shape_full'
                'confounding_texture_full', 'confounding_position_full']
test_levels = ['deconfounding_color_full', 'deconfounding_shape_full'
               'deconfounding_texture_full', 'deconfounding_position_full']
```

For the meta-learning/experimenting version, use the following train levels:

```python
train_levels = [
  'meta_3_easy_color_full', 'meta_3_easy_shape_full',
  'meta_3_easy_texture_full',
  'meta_3_hard1_color_full', 'meta_3_hard1_shape_full',
  'meta_3_hard1_texture_full',
  'meta_3_hard2_color_full', 'meta_3_hard2_shape_full',
  'meta_3_hard2_texture_full',
  'meta_3_hard3_color_full', 'meta_3_hard3_shape_full',
  'meta_3_hard3_texture_full',
]
```


## Citing this work

If you use this work, please cite the associated paper
(https://arxiv.org/abs/2112.03753):

```
@inproceedings{lampinen2022tell,
  title={Tell me why! Explanations support learning relational and causal structure},
  author={Lampinen, Andrew K and Roy, Nicholas A and Dasgupta, Ishita and Chan, Stephanie CY and Tam, Allison C and McClelland, James L and Yan, Chen and Santoro, Adam and Rabinowitz, Neil C and Wang, Jane X and others},
  booktitle={International Conference on Machine Learning},
  year={2022}
}
```

## License and disclaimer

Copyright 2022 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.

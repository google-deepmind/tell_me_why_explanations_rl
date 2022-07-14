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

"""An example that builds and steps a few different environments."""

from tell_me_why_explanations_rl import odd_one_out_environment

train_levels = ['color_full', 'shape_full', 'texture_full', 'position_full']
train_envs = [odd_one_out_environment.from_name(l) for l in train_levels]

# try one out
print(f'Loading environment {train_levels[0]}.')
env = train_envs[0]
timestep = env.reset()
for action in range(8):
  timestep = env.step(action)
print('Environment ran successfully.')

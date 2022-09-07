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

"""The odd-one-out environment classes.

This file builds the dm_env interfaces for the core pycolab games, which
generate the inputs for the agent from the pycolab state representation, and
put everything into a nice dm_env format.
"""
import copy
import functools

from absl import logging

import dm_env

import numpy as np

from pycolab import cropping
from pycolab import storytelling

from tell_me_why_explanations_rl import odd_one_out_environment_core as _env_core

UPSAMPLE_SIZE = 9  # pixels per game square
SCROLL_CROP_SIZE = 11

OBJECT_SHAPES = [
    "triangle", "plus", "inverse_plus", "ex", "inverse_ex", "circle", "tee",
    "upside_down_tee", "h", "u", "upside_down_u",
]
OBJECT_TEXTURES = [
    "solid", "horizontal_stripes", "vertical_stripes",
    "checker", "grid", "noise",
]

COLORS = {
    "red": np.array([255, 0, 0]),
    "green": np.array([0, 255, 0]),
    "blue": np.array([0, 0, 255]),
    "purple": np.array([128, 0, 128]),
    "orange": np.array([255, 165, 0]),
    "yellow": np.array([255, 255, 0]),
    "brown": np.array([128, 64, 0]),
    "pink": np.array([255, 64, 255]),
    "cyan": np.array([0, 255, 255]),
    "dark_green": np.array([0, 100, 0]),
    "dark_red": np.array([100, 0, 0]),
    "dark_blue": np.array([0, 0, 100]),
    "olive": np.array([100, 100, 0]),
    "teal": np.array([0, 100, 100]),
    "lavender": np.array([215, 200, 255]),
    "peach": np.array([255, 210, 170]),
    "rose": np.array([255, 205, 230]),
    "light_green": np.array([200, 255, 200]),
    "light_yellow": np.array([255, 255, 200]),
}


# meta-learning only
EXPERIMENT_REWARD_MULT = 1.  # how big the experimentation trial rewards are
TEST_REWARD_MULT = 10.  # how big the final trial rewards are


def _generate_template(object_name):
  """Generate a template image from an object color + texture + shape string."""
  object_color, object_texture, object_type = object_name.split()
  template = np.zeros((UPSAMPLE_SIZE, UPSAMPLE_SIZE))
  half = UPSAMPLE_SIZE // 2
  if object_type == "triangle":
    for i in range(UPSAMPLE_SIZE):
      for j in range(UPSAMPLE_SIZE):
        if (j <= half and i >= 2 * (half - j)) or (j > half and i >= 2 *
                                                   (j - half)):
          template[i, j] = 1.
  elif object_type == "square":
    template[:, :] = 1.
  elif object_type == "empty_square":
    template[:2, :] = 1.
    template[-2:, :] = 1.
    template[:, :2] = 1.
    template[:, -2:] = 1.
  elif object_type == "plus":
    template[:, half - 1:half + 2] = 1.
    template[half - 1:half + 2, :] = 1.
  elif object_type == "inverse_plus":
    template[:, :] = 1.
    template[:, half - 1:half + 2] = 0.
    template[half - 1:half + 2, :] = 0.
  elif object_type == "ex":
    for i in range(UPSAMPLE_SIZE):
      for j in range(UPSAMPLE_SIZE):
        if abs(i - j) <= 1 or abs(UPSAMPLE_SIZE - 1 - j - i) <= 1:
          template[i, j] = 1.
  elif object_type == "inverse_ex":
    for i in range(UPSAMPLE_SIZE):
      for j in range(UPSAMPLE_SIZE):
        if not (abs(i - j) <= 1 or abs(UPSAMPLE_SIZE - 1 - j - i) <= 1):
          template[i, j] = 1.
  elif object_type == "circle":
    for i in range(UPSAMPLE_SIZE):
      for j in range(UPSAMPLE_SIZE):
        if (i - half)**2 + (j - half)**2 <= half**2:
          template[i, j] = 1.
  elif object_type == "empty_circle":
    for i in range(UPSAMPLE_SIZE):
      for j in range(UPSAMPLE_SIZE):
        if abs((i - half)**2 + (j - half)**2 - half**2) < 6:
          template[i, j] = 1.
  elif object_type == "tee":
    template[:, half - 1:half + 2] = 1.
    template[:3, :] = 1.
  elif object_type == "upside_down_tee":
    template[:, half - 1:half + 2] = 1.
    template[-3:, :] = 1.
  elif object_type == "h":
    template[:, :3] = 1.
    template[:, -3:] = 1.
    template[half - 1:half + 2, :] = 1.
  elif object_type == "u":
    template[:, :3] = 1.
    template[:, -3:] = 1.
    template[-3:, :] = 1.
  elif object_type == "upside_down_u":
    template[:, :3] = 1.
    template[:, -3:] = 1.
    template[:3, :] = 1.
  elif object_type == "vertical_stripes":
    for j in range(half + UPSAMPLE_SIZE % 2):
      template[:, 2*j] = 1.
  elif object_type == "horizontal_stripes":
    for i in range(half + UPSAMPLE_SIZE % 2):
      template[2*i, :] = 1.
  else:
    raise ValueError("Unknown object: {}".format(object_type))

  texture = np.ones_like(template)
  offset = UPSAMPLE_SIZE % 2
  if object_texture == "horizontal_stripes":
    texture[offset::2, :] = 0.
  elif object_texture == "vertical_stripes":
    texture[:, offset::2] = 0.
  elif object_texture == "grid":
    texture[:, :] = 0.
    texture[offset::2, :] = 1.
    texture[:, offset::2] = 1.
  elif object_texture == "checker":
    texture[offset::2, ::2] = 0.
    texture[::2, offset::2] = 0.
  elif object_texture == "noise":
    texture = np.random.binomial(1, 0.66, texture.shape)
  elif object_texture != "solid":
    raise ValueError("Unknown texture: {}".format(object_texture))

  if object_color not in COLORS:
    raise ValueError("Unknown color: {}".format(object_color))

  template = template * texture
  template = np.tensordot(template, COLORS[object_color], axes=0)

  return template


# Agent, wall, and floor templates
_CHAR_TO_TEMPLATE_BASE = {
    _env_core.AGENT_CHAR:
        np.tensordot(
            np.ones([UPSAMPLE_SIZE, UPSAMPLE_SIZE]),
            np.array([255, 255, 255]),
            axes=0),
    _env_core.WALL_CHAR:
        np.tensordot(
            np.ones([UPSAMPLE_SIZE, UPSAMPLE_SIZE]),
            np.array([40, 40, 40]),
            axes=0),
    ## FLOOR_CHAR is now skipped, for efficiency
}


class OddOneOutEnvironment(dm_env.Environment):
  """A dm_env version of Odd One Out, including pixel observations etc."""

  def __init__(
      self,
      game_factory,
      max_steps=_env_core.EPISODE_LENGTH,
      rng=None):
    """Construct a dm_env-compatible wrapper for pycolab games for agent use.

    This class inherits from dm_env and has all the expected methods and specs.
    It also renders the game from ascii art to pixel observations.

    Args:
      game_factory: A function that when called returns a new pycolab core game.
      max_steps: The maximum number of steps to allow in an episode, after which
          it will terminate.
      rng: An optional numpy Random Generator, to set a fixed seed use e.g.
          `rng=np.random.default_rng(seed=...)`
    """
    self._game_factory = game_factory
    self._max_steps = max_steps

    # internal state
    if rng is None:
      rng = np.random.default_rng()
    self._rng = rng
    self._current_game = None       # Current pycolab game instance.
    self._state = None              # Current game step state.
    self._game_over = None          # Whether the game has ended.
    self._char_to_template = None   # Mapping of chars to images.
    self._cue_template = None

    # rendering tools
    self._cropper = cropping.ScrollingCropper(
        rows=SCROLL_CROP_SIZE, cols=SCROLL_CROP_SIZE,
        to_track=[_env_core.AGENT_CHAR], pad_char=_env_core.FLOOR_CHAR,
        scroll_margins=(None, None))

  def _render_observation(self, observation):
    """Renders from raw pycolab image observation to agent-usable pixel ones."""
    observation = self._cropper.crop(observation)
    obs_rows, obs_cols = observation.board.shape
    image = np.zeros([obs_rows * UPSAMPLE_SIZE, obs_cols * UPSAMPLE_SIZE, 3],
                     dtype=np.float32)
    for i in range(obs_rows):
      for j in range(obs_cols):
        this_char = chr(observation.board[i, j])
        if this_char != _env_core.FLOOR_CHAR:
          image[
              i * UPSAMPLE_SIZE:(i + 1) * UPSAMPLE_SIZE, j *
              UPSAMPLE_SIZE:(j + 1) * UPSAMPLE_SIZE] = self._char_to_template[
                  this_char]
    image /= 255.
    # explanation observation
    explanation = np.array(self._current_game.the_plot["explanation_string"])
    return (image, explanation)

  def _update_char_to_template(self):
    self._char_to_template = {
        k: _generate_template(v) for k, v in self._current_game.the_plot[
            "char_to_color_shape"].items()}
    self._char_to_template.update(_CHAR_TO_TEMPLATE_BASE)

  def reset(self):
    """Start a new episode."""
    # clear old state
    self._state = None
    self._current_game = None
    self._char_to_template = None
    self._game_over = None
    # Build a new game and retrieve its first set of state/reward/discount.
    self._current_game = self._game_factory()
    # set up rendering, cropping, and state for current game
    self._update_char_to_template()

    self._cropper.set_engine(self._current_game)
    self._state = dm_env.StepType.FIRST
    # let's go!
    observation, _, _ = self._current_game.its_showtime()
    observation = self._render_observation(observation)
    return dm_env.restart(observation)

  def step(self, action):
    """Apply action, step the world forward, and return observations."""
    # If needed, reset and start new episode.
    if self._current_game is None or self._state.last():
      return self.reset()

    # Execute the action in pycolab.
    observation, reward, discount = self._current_game.play(action)
    if self._current_game.the_plot["char_to_color_shape_updated"]:
      self._update_char_to_template()

    self._game_over = self._is_game_over()
    reward = reward if reward is not None else 0.
    observation = self._render_observation(observation)

    # Check the current status of the game.
    if self._game_over:
      self._state = dm_env.StepType.LAST
    else:
      self._state = dm_env.StepType.MID

    return dm_env.TimeStep(
        step_type=self._state,
        reward=reward,
        discount=discount,
        observation=observation)

  def observation_spec(self):
    image_shape = (SCROLL_CROP_SIZE * UPSAMPLE_SIZE,
                   SCROLL_CROP_SIZE * UPSAMPLE_SIZE,
                   3)
    return (
        # vision
        dm_env.specs.Array(
            shape=image_shape, dtype=np.float32, name="image"),
        # explanation
        dm_env.specs.Array(
            shape=[], dtype=str, name="explanation"),
        )

  def action_spec(self):
    return dm_env.specs.BoundedArray(
        shape=[], dtype="int32",
        minimum=0, maximum=7,
        name="grid_actions")

  def _is_game_over(self):
    """Returns whether it is game over, either from the engine or timeout."""
    return (self._current_game.game_over or
            (self._current_game.the_plot.frame >= self._max_steps))


class MetaOddOneOutEnvironment(OddOneOutEnvironment):
  """For metalearning version, tweaks to actions + observations."""

  def _render_observation(self, observation):
    """Renders from raw pycolab image observation to agent-usable pixel ones."""
    base_obs = super()._render_observation(observation)
    instruction = np.array(self._current_game.the_plot["instruction_string"])
    return (*base_obs, instruction)

  def observation_spec(self):
    base_spec = super().observation_spec()
    return (
        *base_spec,
        # instruction
        dm_env.specs.Array(
            shape=[], dtype=str, name="explanation"),
        )

  def action_spec(self):
    return dm_env.specs.BoundedArray(
        shape=[], dtype="int32",
        minimum=0, maximum=10,
        name="grid_and_transform_actions")


def builder(concept_type="shape", explain="full", rng=None):
  """Build a game factory and dm_env wrapper around the basic odd one out tasks.

  Args:
    concept_type: concept type, one of ["color", "shape", "texture", "position"]
    explain: How to explain, one of ["full", "properties", "reward", or "none"]
    rng: An optional numpy Random Generator, to set a fixed seed use e.g.
        `rng=np.random.default_rng(seed=...)`

  Returns:
    OddOneOutEnvironment object for the specified level.
  """
  if concept_type not in ["shape", "color", "texture", "position"]:
    raise NotImplementedError(
        "Level construction with concepts other than shape, color, texture, or "
        "position is not yet supported.")

  if rng is None:
    rng = np.random.default_rng()

  position_types = list(_env_core.OBJECT_POSITIONS.keys())
  positions = copy.deepcopy(_env_core.OBJECT_POSITIONS)
  colors = list(COLORS)
  shapes = OBJECT_SHAPES.copy()
  textures = OBJECT_TEXTURES.copy()

  def _game_factory():
    """Samples pairing and positions, returns a game."""
    target_object_index = rng.integers(4)
    rng.shuffle(position_types)
    for v in positions.values():
      rng.shuffle(v)
    rng.shuffle(colors)
    rng.shuffle(textures)
    rng.shuffle(shapes)
    if concept_type == "color":
      these_colors = [colors[0]] * 4
      these_colors[target_object_index] = colors[1]
    else:
      these_colors = [colors[0]] * 2 + [colors[1]] * 2
      rng.shuffle(these_colors)

    if concept_type == "texture":
      these_textures = [textures[0]] * 4
      these_textures[target_object_index] = textures[1]
    else:
      these_textures = [textures[0]] * 2 + [textures[1]] * 2
      rng.shuffle(these_textures)

    if concept_type == "shape":
      these_shapes = [shapes[0]] * 4
      these_shapes[target_object_index] = shapes[1]
    else:
      these_shapes = [shapes[0]] * 2 + [shapes[1]] * 2
      rng.shuffle(these_shapes)

    if concept_type == "position":
      these_position_types = [position_types[0]] * 4
      these_position_types[target_object_index] = position_types[1]
    else:
      these_position_types = [position_types[0]] * 2 + [position_types[1]] * 2
      rng.shuffle(these_position_types)

    object_properties = []
    for object_i in range(4):
      if object_i == target_object_index:
        value = 1.
      else:
        value = 0.
      # choose a position of this type
      position = positions[these_position_types[object_i]][object_i]
      object_properties.append(
          _env_core.ObjectProperties(
              character=_env_core.POSSIBLE_OBJECT_CHARS[object_i],
              position=position,
              position_type=these_position_types[object_i],
              shape=these_shapes[object_i],
              color=these_colors[object_i],
              texture=these_textures[object_i],
              value=value))

    logging.info("Making level with object_properties: %s",
                 object_properties)

    return _env_core.make_game(
        object_properties=object_properties,
        concept_type=concept_type,
        explain=explain, rng=rng)

  return OddOneOutEnvironment(
      game_factory=_game_factory, rng=rng)


def metalearning_builder(num_trials_before_test=3, intervention_type="easy",
                         concept_type="shape", explain="full", rng=None):
  """Builds a meta-learning environment with several experiment trials + test.

  Args:
    num_trials_before_test: how many "experiment" trials to have, for the
      agent to figure out the task
    intervention_type: whether the intervention levels are easy (all objects the
      same, change one and take it) or hard{1,2,3} (all objects differ in pairs,
      along 1, 2, or 3 dimension, so if you change one you have to take its pair
      that you've made the odd one out).
    concept_type: concept type, see _env_core.MetaOddOneOutEnvironment.
    explain: How to explain, see _env_core.MetaOddOneOutEnvironment.
    rng: An optional numpy Random Generator, to set a fixed seed use e.g.
        `rng=np.random.default_rng(seed=...)`

  Returns:
    MetaOddOneOutEnvironment object for the specified level.
  """
  if concept_type not in ["shape", "color", "texture"]:
    raise NotImplementedError(
        "The currently supported concepts are shape, color, or texture.")
  if rng is None:
    rng = np.random.default_rng()

  position_types = list(_env_core.OBJECT_POSITIONS.keys())
  positions = copy.deepcopy(_env_core.OBJECT_POSITIONS)
  colors = list(COLORS.keys())
  shapes = OBJECT_SHAPES.copy()
  textures = OBJECT_TEXTURES.copy()

  def _level_factory(level_type="regular",
                     level_args=None):
    if level_args is None:
      level_args = {}
    if level_type == "deconfounded":
      target_indices = rng.permutation(4)
      if concept_type == "color":
        target_object_index = target_indices[0]
      elif concept_type == "texture":
        target_object_index = target_indices[1]
      elif concept_type == "shape":
        target_object_index = target_indices[2]
      else:
        raise ValueError()
    else:
      target_object_index = rng.integers(4)
    rng.shuffle(position_types)
    for v in positions.values():
      rng.shuffle(v)
    rng.shuffle(colors)
    rng.shuffle(textures)
    rng.shuffle(shapes)
    additional_extant_properties = {}

    if level_type[:-1] == "intervention_hard":
      num_hard_dimensions = int(level_type[-1])
      assert num_hard_dimensions in [1, 2, 3]
      hard_dimensions = ["color", "texture", "shape"]
      rng.shuffle(hard_dimensions)
      hard_dimensions = hard_dimensions[:num_hard_dimensions]
    else:
      num_hard_dimensions = 0
      hard_dimensions = []

    if "intervention" in level_type:
      if "color" in hard_dimensions:
        these_colors = [colors[0]] * 2 + [colors[1]] * 2
        rng.shuffle(these_colors)
      else:
        these_colors = [colors[0]] * 4
        additional_extant_properties["color"] = [colors[1]]
    elif level_type == "deconfounded":
      these_colors = [colors[0]] * 4
      these_colors[target_indices[0]] = colors[1]
    elif concept_type == "color":
      these_colors = [colors[0]] * 4
      these_colors[target_object_index] = colors[1]
    else:
      these_colors = [colors[0]] * 2 + [colors[1]] * 2
      rng.shuffle(these_colors)

    if "intervention" in level_type:
      if "texture" in hard_dimensions:
        these_textures = [textures[0]] * 2 + [textures[1]] * 2
        rng.shuffle(these_textures)
      else:
        these_textures = [textures[0]] * 4
        additional_extant_properties["texture"] = [textures[1]]
    elif level_type == "deconfounded":
      these_textures = [textures[0]] * 4
      these_textures[target_indices[1]] = textures[1]
    elif concept_type == "texture":
      these_textures = [textures[0]] * 4
      these_textures[target_object_index] = textures[1]
    else:
      these_textures = [textures[0]] * 2 + [textures[1]] * 2
      rng.shuffle(these_textures)

    if "intervention" in level_type:
      if "shape" in hard_dimensions:
        these_shapes = [shapes[0]] * 2 + [shapes[1]] * 2
        rng.shuffle(these_shapes)
      else:
        these_shapes = [shapes[0]] * 4
        additional_extant_properties["shape"] = [shapes[1]]
    elif level_type == "deconfounded":
      these_shapes = [shapes[0]] * 4
      these_shapes[target_indices[2]] = shapes[1]
    elif concept_type == "shape":
      these_shapes = [shapes[0]] * 4
      these_shapes[target_object_index] = shapes[1]
    else:
      these_shapes = [shapes[0]] * 2 + [shapes[1]] * 2
      rng.shuffle(these_shapes)

    these_position_types = [position_types[0]] * 4

    object_properties = []
    for object_i in range(4):
      if object_i == target_object_index and "intervention" not in level_type:
        value = 1.
      else:
        value = 0.
      # choose a (distinct) position of this type
      position = positions[these_position_types[object_i]][object_i]
      object_properties.append(
          _env_core.ObjectProperties(
              character=_env_core.POSSIBLE_OBJECT_CHARS[object_i],
              position=position,
              position_type=these_position_types[object_i],
              shape=these_shapes[object_i],
              color=these_colors[object_i],
              texture=these_textures[object_i],
              value=value))

    logging.info("Making metalearning level with level_type: %s and "
                 "object_properties: %s",
                 level_type, object_properties)

    return _env_core.make_metalearning_game(
        object_properties=object_properties,
        concept_type=concept_type,
        explain=explain,
        additional_extant_properties=additional_extant_properties,
        rng=rng,
        **level_args)

  intervention_level = "intervention_" + intervention_type
  def _progressive_game_factory():
    level_constructors = {}
    for i in range(num_trials_before_test):
      level_constructors[i] = functools.partial(
          _level_factory,
          intervention_level,
          dict(
              transformations_allowed=1,
              value_multiplier=EXPERIMENT_REWARD_MULT,
              next_progressive_level=i + 1))

    i = num_trials_before_test
    level_constructors[i] = functools.partial(
        _level_factory,
        "deconfounded",
        dict(
            transformations_allowed=0,
            value_multiplier=TEST_REWARD_MULT,
            next_progressive_level=None))

    this_story = storytelling.Story(
        chapters=level_constructors,
        first_chapter=0)
    return this_story

  num_sub_episodes = num_trials_before_test + 1
  return MetaOddOneOutEnvironment(
      game_factory=_progressive_game_factory,
      max_steps=num_sub_episodes * _env_core.EPISODE_LENGTH,
      rng=rng)


def confounding_builder(confounding, concept_type, explain="none", rng=None):
  """Build a game factory and dm_env wrapper around the (de)confounding tasks.

  Args:
    confounding: one of ["confounded", "deconfounded"].
    concept_type: concept type, one of ["color", "shape", "texture"].
    explain: How to explain, one of ["full", "properties", "reward", or "none"].
    rng: An optional numpy Random Generator, to set a fixed seed use e.g.
        `rng=np.random.default_rng(seed=...)`

  Returns:
    OddOneOutEnvironment object for the specified level.
  """
  if rng is None:
    rng = np.random.default_rng()
  position_types = list(_env_core.OBJECT_POSITIONS.keys())
  positions = copy.deepcopy(_env_core.OBJECT_POSITIONS)
  colors = list(COLORS.keys())
  shapes = OBJECT_SHAPES.copy()
  textures = OBJECT_TEXTURES.copy()

  def _game_factory():
    target_indices = rng.permutation(4)
    # fixed assignment of indices (in this ordering) to attributes,
    # for deconfounded version.
    if concept_type == "color":
      target_object_index = target_indices[0]
    elif concept_type == "texture":
      target_object_index = target_indices[1]
    elif concept_type == "shape":
      target_object_index = target_indices[2]
    elif concept_type == "position":
      target_object_index = target_indices[3]
    rng.shuffle(position_types)
    for v in positions.values():
      rng.shuffle(v)
    rng.shuffle(colors)
    rng.shuffle(textures)
    rng.shuffle(shapes)
    these_colors = [colors[0]] * 4
    if concept_type == "color" or confounding == "confounded":
      these_colors[target_object_index] = colors[1]
    else:  # confounding == "deconfounded"
      these_colors[target_indices[0]] = colors[1]

    these_textures = [textures[0]] * 4
    if concept_type == "texture" or confounding == "confounded":
      these_textures[target_object_index] = textures[1]
    else:
      these_textures[target_indices[1]] = textures[1]

    these_shapes = [shapes[0]] * 4
    if concept_type == "shape" or confounding == "confounded":
      these_shapes[target_object_index] = shapes[1]
    else:
      these_shapes[target_indices[2]] = shapes[1]

    these_position_types = [position_types[0]] * 4

    object_properties = []
    for object_i in range(4):
      if object_i == target_object_index:
        value = 1.
      else:
        value = 0.
      # choose a position of this type
      position = positions[these_position_types[object_i]][object_i]
      object_properties.append(
          _env_core.ObjectProperties(
              character=_env_core.POSSIBLE_OBJECT_CHARS[object_i],
              position=position,
              position_type=these_position_types[object_i],
              shape=these_shapes[object_i],
              color=these_colors[object_i],
              texture=these_textures[object_i],
              value=value))

    logging.info("Making level with confounding: %s and "
                 "object_properties: %s",
                 confounding, object_properties)

    return _env_core.make_game(
        object_properties=object_properties,
        concept_type=concept_type,
        explain=explain,
        explain_only_concept_type=True)

  return OddOneOutEnvironment(
      game_factory=_game_factory, rng=rng)


def from_name(level_name):
  """Simplifies building basic levels from fixed defs."""
  if level_name[:4] == "meta":
    (_, num_experiment_trials, intervention_type,
     concept_type, explain) = level_name.split("_")
    num_experiment_trials = int(num_experiment_trials)
    return metalearning_builder(
        num_trials_before_test=num_experiment_trials,
        intervention_type=intervention_type,
        concept_type=concept_type, explain=explain)
  elif "confound" in level_name:
    confounding, concept_type, explain = level_name.split("_")
    return confounding_builder(confounding, concept_type, explain)
  else:
    concept_type, explain = level_name.split("_")
    return builder(concept_type, explain)

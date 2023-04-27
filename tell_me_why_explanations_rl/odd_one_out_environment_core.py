# Copyright 2023 DeepMind Technologies Limited
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

"""A base pycolab environment for finding the object that's an odd one out.

The agent will be placed in a room with four objects. These objects have
different colors, shapes, and possible positions. Two values of each attribute
will appear in any given trial. For example, there might be red and blue objects
triangles and squares, and objects either in the corner or center. Two of these
attributes will be evenly distributed across the objects, so that half the
objects have each value, but one will be unevenly split. For example, there
might be two red and two blue objects, two triangles and two squares, but three
objects in the center and only one in the corner. The agent will be rewarded for
touching the object that's the odd one out.

For the meta-learning task, the code in this file mostly corresponds to a single
trial; the linking across trials is handled in the MetaOddOneOutEnvironment
class in odd_one_out_environment.py.
"""
import curses

import dataclasses
import enum
from typing import Tuple

from absl import app
from absl import logging

import numpy as np

from pycolab import ascii_art
from pycolab import human_ui
from pycolab import things as plab_things
from pycolab.prefab_parts import sprites as prefab_sprites


ROOM_SIZE = (11, 11)  # one square around edge will be wall.
OBJECT_POSITIONS = {
    "in_corner": [(1, 1), (1, 9), (9, 1), (9, 9)],
    "against_wall_x": [(1, 4), (1, 5), (1, 6), (9, 4), (9, 5), (9, 6)],
    "against_wall_y": [(4, 1), (5, 1), (6, 1), (4, 9), (5, 9), (6, 9)],
    "in_center": [(4, 4), (4, 5), (4, 6),
                  (5, 4), (5, 5), (5, 6),
                  (6, 4), (6, 5), (6, 6)]}
AGENT_CHAR = "A"
WALL_CHAR = "#"
FLOOR_CHAR = " "
RESERVED_CHARS = [AGENT_CHAR, WALL_CHAR, FLOOR_CHAR]
POSSIBLE_OBJECT_CHARS = [
    chr(i) for i in range(65, 91) if chr(i) not in RESERVED_CHARS
]

EXPLAIN_PHASE_LENGTH = 16
EPISODE_LENGTH = 128

CORRECT_REWARD = 1.
INCORRECT_REWARD = 0.


META_BETWEEN_TRIAL_CLEANUP_KEYS = (  # plot items to reinitialize between trials
    "next_progressive_level", "explanation_string", "instruction_string",
    "char_to_color_shape", "char_to_color_shape_updated",
    "transformations_allowed", "transformations_happening_now",
    "extant_attributes", "concept_type", "explain", "termination_countdown",
    "value_multiplier",
)


class ACTIONS(enum.IntEnum):
  """The possible actions the agent can take."""
  # movement directions
  MOVE_N = 0
  MOVE_NE = 1
  MOVE_E = 2
  MOVE_SE = 3
  MOVE_S = 4
  MOVE_SW = 5
  MOVE_W = 6
  MOVE_NW = 7
  # transformations (used for meta environment only)
  TRANSFORM_COLOR = 8
  TRANSFORM_TEXTURE = 9
  TRANSFORM_SHAPE = 10

TRANSFORM_ACTIONS = (
    ACTIONS.TRANSFORM_COLOR, ACTIONS.TRANSFORM_TEXTURE, ACTIONS.TRANSFORM_SHAPE)


def terminate_episode_cleanup(the_plot):
  """Cleans up between trials in meta-learning setting."""
  if the_plot["next_progressive_level"] is not None:
    the_plot.next_chapter = the_plot["next_progressive_level"]
    logging.info("Progressing to next level! %i", the_plot.next_chapter)
    # don't carry these keys over, will be reset when the new game is built
    for k in META_BETWEEN_TRIAL_CLEANUP_KEYS:
      del the_plot[k]


@dataclasses.dataclass
class ObjectProperties:
  """Class for holding the properties of objects while building."""
  character: str
  position: Tuple[int, int]
  position_type: str
  shape: str
  color: str
  texture: str
  value: float


class ObjectDrape(plab_things.Drape):
  """A `Drape` for objects in the room.

  See parent class for details of Drapes. These drapes handle logic of providing
  explanations to the agent, and handle rewards etc. if agent moves onto one.
  """

  def __init__(self, curtain, character, object_properties, properties_string,
               explanation_string):
    assert character == object_properties.character
    super(ObjectDrape, self).__init__(curtain, object_properties.character)
    self.color = object_properties.color
    self.texture = object_properties.texture
    self.shape = object_properties.shape
    self.properties_string = properties_string
    self.explanation_string = explanation_string
    self.value = object_properties.value
    self.position = object_properties.position
    self.agent_is_adjacent = False

  def _handle_player_touch(self, the_plot):
    """What happens if player moves onto this object."""
    if the_plot["termination_countdown"] is not None:
      if the_plot["termination_countdown"] < EXPLAIN_PHASE_LENGTH - 2:
        # touched something else already, and time passed since, terminate.
        the_plot.terminate_episode()
    else:  # touched for the very first time!
      the_plot.add_reward(self.value)
      the_plot["termination_countdown"] = EXPLAIN_PHASE_LENGTH
      the_plot["explanation_string"] = self.explanation_string

  def _handle_adjacency(self, the_plot, adjacent, *args):
    """What happens if player is adjacent to this object."""
    if adjacent:
      if the_plot["termination_countdown"] is None:
        the_plot["explanation_string"] = self.properties_string
    elif self.agent_is_adjacent:
      # no longer adjacent, but was, reset instruction if needed
      self.agent_is_adjacent = False
      if the_plot["explanation_string"] == self.properties_string:
        the_plot["explanation_string"] = ""

  def update(self, actions, board, layers, backdrop, things, the_plot):
    """Update state given player actions, etc. Player updates earlier."""
    rows, cols = things[AGENT_CHAR].position
    if self.curtain[(rows, cols)]:
      self._handle_player_touch(the_plot)
    else:
      # is character adjacent to object?
      adjacent = False
      poss_rows = range(rows - 1, rows + 2)
      poss_cols = range(cols - 1, cols + 2)
      for x in poss_rows:
        for y in poss_cols:
          possible_position = (x, y)
          if self.curtain[possible_position]:
            adjacent = True
            break
      self._handle_adjacency(the_plot, adjacent, actions, things)


class MetaObjectDrape(ObjectDrape):
  """A `Drape` for objects in the meta-learning version."""

  def _set_value_and_explanations(self, things, the_plot):
    """Update value and explanations (e.g., after any transformations)."""
    self.value = CORRECT_REWARD
    for k, thing in things.items():
      if k not in [self.character, AGENT_CHAR]:
        # if matches along relevant dimension, then not odd one out
        if ((the_plot["concept_type"] == "color" and
             thing.color == self.color) or
            (the_plot["concept_type"] == "texture" and
             thing.texture == self.texture) or
            (the_plot["concept_type"] == "shape" and
             thing.shape == self.shape)):
          self.value = INCORRECT_REWARD
          break
    # same with explanations
    if self.explanation_string:
      if self.value == CORRECT_REWARD:
        explanation_string = ["Correct the concept is"]
        explanation_string += [the_plot["concept_type"]]
        explanation_string += ["and it is uniquely"]
      else:
        explanation_string = ["Incorrect the concept is"]
        explanation_string += [the_plot["concept_type"]]
        explanation_string += ["and other objects are"]
      if the_plot["concept_type"] == "color":
        explanation_string.append(self.color)
      elif the_plot["concept_type"] == "shape":
        explanation_string.append(self.shape)
      elif the_plot["concept_type"] == "texture":
        explanation_string.append(self.texture)
      self.explanation_string = " ".join(explanation_string)
    if self.properties_string:
      self.properties_string = " ".join([
          "This is a", self.color, self.texture, self.shape])

  def _handle_player_touch(self, the_plot):
    if the_plot["termination_countdown"] is not None:
      if the_plot["termination_countdown"] < EXPLAIN_PHASE_LENGTH - 2:
        # touched something else already, and time passed since, terminate.
        terminate_episode_cleanup(the_plot)
        the_plot.terminate_episode()
        return
    else:
      the_plot.add_reward(self.value * the_plot["value_multiplier"])
      the_plot["termination_countdown"] = EXPLAIN_PHASE_LENGTH
      the_plot["explanation_string"] = self.explanation_string
      the_plot["instruction_string"] = ""

  def _handle_adjacency(self, the_plot, adjacent, actions, things):
    if adjacent:
      if the_plot["transformations_happening_now"] and actions in [8, 9, 10]:
        print("transforming adjacent")
        original = the_plot["char_to_color_shape"][self.character]
        original = original.split()
        updated = None
        if actions == ACTIONS.TRANSFORM_COLOR:
          for other_color in the_plot["extant_attributes"]["color"]:
            if other_color != self.color:
              logging.info("Transforming: %s -> %s", self.color, other_color)
              self.color = other_color
              updated = [other_color] +  original[1:]
              break
        elif actions == ACTIONS.TRANSFORM_TEXTURE:  # transform texture
          for other_texture in the_plot["extant_attributes"]["texture"]:
            if other_texture != self.texture:
              logging.info(
                  "Transforming: %s -> %s", self.texture, other_texture)
              self.texture = other_texture
              updated = [original[0], other_texture, original[2]]
              break
        elif actions == ACTIONS.TRANSFORM_SHAPE:  # transform shape
          for other_shape in the_plot["extant_attributes"]["shape"]:
            if other_shape != self.shape:
              logging.info("Transforming: %s -> %s", self.shape, other_shape)
              self.shape = other_shape
              updated = original[:2] + [other_shape]
              break
        updated = " ".join(updated)
        the_plot["char_to_color_shape"][self.character] = updated
        the_plot["char_to_color_shape_updated"] = True

      # update value etc. anytime player is adjacent, when it matters...
      self._set_value_and_explanations(things, the_plot)

      if (the_plot["termination_countdown"] is None and
          the_plot["explanation_string"][:15] != "You transformed"):
        the_plot["explanation_string"] = self.properties_string
    elif self.agent_is_adjacent:
      # no longer adjacent, but was, reset instruction if needed
      self.agent_is_adjacent = False
      if the_plot["explanation_string"] == self.properties_string:
        the_plot["explanation_string"] = ""

  def update(self, actions, board, layers, backdrop, things, the_plot):
    if "char_to_color_shape" not in the_plot:  # trial over, nothing left to do!
      return
    return super().update(actions, board, layers, backdrop, things, the_plot)


class PlayerSprite(prefab_sprites.MazeWalker):
  """The player character.

  Player character, moves around and handles some game logic. See parent class
  for further details.
  """

  def __init__(self, corner, position, character):
    super(PlayerSprite, self).__init__(
        corner, position, character, impassable="#")
    self.start_position = position

  def _terminate_episode(self, the_plot):
    the_plot.terminate_episode()

  def update(self, actions, board, layers, backdrop, things, the_plot):
    """Update self and game state given an action."""
    # basic movement
    if actions == ACTIONS.MOVE_N:
      self._north(board, the_plot)
    elif actions == ACTIONS.MOVE_NE:
      self._northeast(board, the_plot)
    elif actions == ACTIONS.MOVE_E:
      self._east(board, the_plot)
    elif actions == ACTIONS.MOVE_SE:
      self._southeast(board, the_plot)
    elif actions == ACTIONS.MOVE_S:
      self._south(board, the_plot)
    elif actions == ACTIONS.MOVE_SW:
      self._southwest(board, the_plot)
    elif actions == ACTIONS.MOVE_W:
      self._west(board, the_plot)
    elif actions == ACTIONS.MOVE_NW:
      self._northwest(board, the_plot)
    # game logic
    if the_plot["termination_countdown"] is not None:
      if the_plot["termination_countdown"] == 0:
        self._terminate_episode(the_plot)
      else:
        the_plot["termination_countdown"] -= 1


class MetaPlayerSprite(PlayerSprite):
  """The player for the meta-learning tasks."""

  def _terminate_episode(self, the_plot):
    terminate_episode_cleanup(the_plot)
    the_plot.terminate_episode()

  def update(self, actions, board, layers, backdrop, things, the_plot):
    if actions in TRANSFORM_ACTIONS and the_plot["transformations_allowed"] > 0:
      the_plot["transformations_allowed"] -= 1
      the_plot["transformations_happening_now"] = True
    else:
      the_plot["transformations_happening_now"] = False
      if the_plot["explanation_string"][:15] == "You transformed":
        the_plot["explanation_string"] = ""
    super().update(actions, board, layers, backdrop, things, the_plot)


def _generate_level_layout(object_properties, agent_start):
  """Generates pycolab-style ascii map containing room, objects, and agent."""
  level_layout = np.array([[FLOOR_CHAR] * ROOM_SIZE[1]] * ROOM_SIZE[0])
  # insert walls
  level_layout[0, :] = WALL_CHAR
  level_layout[-1, :] = WALL_CHAR
  level_layout[:, 0] = WALL_CHAR
  level_layout[:, -1] = WALL_CHAR
  # add agent and objects
  level_layout[agent_start] = AGENT_CHAR
  for obj in object_properties:
    level_layout[obj.position] = obj.character
  # convert to pycolab's ascii format
  level_layout = ["".join(x) for x in level_layout.tolist()]
  return level_layout


def make_game(object_properties, concept_type, explain="full",
              agent_start=None, explain_only_concept_type=False,
              rng=None):
  """Makes a basic pycolab odd-one-out game.

  Args:
    object_properties: list of ObjectProperties for defining objects in level.
    concept_type: one of ["position", "color", "texture" "shape"], indicating
      which attribute has the odd-one-out.
    explain: One of "full" "reward", "properties" or "none." If none, no
      explanation. If "reward" the explanation describes whether the answer was
      correct or incorrect, and the features that show it. If "properties", will
      identify the properties of objects when adjacent to them. If "full", gives
      both properties + reward.
    agent_start: Optional agent start position (mostly for testing).
    explain_only_concept_type: explain only the single dimension corresponding
      to concept_type; used for the confounding experiments only.
    rng: An optional numpy Random Generator for choosing agent_start (if not
      set), to set a fixed seed use e.g. `rng=np.random.default_rng(seed=...)`

  Returns:
    this_game: Pycolab engine running the specified game.
  """
  if rng is None:
    rng = np.random.default_rng()

  char_to_color_shape = []
  drape_creators = {}
  forbidden_locations = []
  for obj in object_properties:
    # can't have player starting here
    forbidden_locations.append(obj.position)
    # instruction
    if explain not in ["none", "full", "reward", "properties"]:
      raise ValueError("Unrecognized explanation type: {}".format(explain))
    if explain in ["full", "properties"]:
      if explain_only_concept_type:
        properties_string = "This is a "
        if concept_type == "color":
          properties_string += obj.color
        elif concept_type == "texture":
          properties_string += obj.texture
        elif concept_type == "shape":
          properties_string += obj.shape
        elif concept_type == "position":
          properties_string += obj.position_type
      else:
        properties_string = " ".join([
            "This is a", obj.color, obj.texture, obj.shape, obj.position_type])
    else:
      properties_string = ""
    explanation_string = ""
    if explain in ["full", "reward"]:
      if obj.value > 0.:
        explanation_string = ["Correct it is uniquely"]
        if concept_type == "position":
          explanation_string.append(obj.position_type)
        elif concept_type == "color":
          explanation_string.append(obj.color)
        elif concept_type == "shape":
          explanation_string.append(obj.shape)
        elif concept_type == "texture":
          explanation_string.append(obj.texture)
      else:
        if explain_only_concept_type:
          explanation_string = ["Incorrect other objects are"]
          if concept_type == "position":
            explanation_string.append(obj.position_type)
          elif concept_type == "color":
            explanation_string.append(obj.color)
          elif concept_type == "shape":
            explanation_string.append(obj.shape)
          elif concept_type == "texture":
            explanation_string.append(obj.texture)
        else:
          explanation_string = [
              "Incorrect other objects are",
              obj.color, obj.texture, obj.shape, "or", obj.position_type]
      explanation_string = " ".join(explanation_string)

    # create object builders
    drape_creators[obj.character] = ascii_art.Partial(
        ObjectDrape, object_properties=obj,
        properties_string=properties_string,
        explanation_string=explanation_string)
    char_to_color_shape.append(
        (obj.character, " ".join((obj.color, obj.texture, obj.shape))))

  # set up agent start
  if agent_start is None:
    poss_starts = []
    for x in range(1, 10):
      for y in range(1, 10):
        if (x, y) not in forbidden_locations:
          poss_starts.append((x, y))
    agent_start = poss_starts[
        rng.integers(len(poss_starts))]
  sprites = {AGENT_CHAR: PlayerSprite}

  # generate level and game
  level_layout = _generate_level_layout(object_properties, agent_start)
  this_game = ascii_art.ascii_art_to_game(
      art=level_layout,
      what_lies_beneath=" ",
      sprites=sprites,
      drapes=drape_creators,
      update_schedule=[[AGENT_CHAR],
                       [obj.character for obj in object_properties]])

  # update necessary plot information
  this_game.the_plot["explanation_string"] = ""
  this_game.the_plot["instruction_string"] = ""  # only used in meta case
  this_game.the_plot["char_to_color_shape"] = dict(char_to_color_shape)
  this_game.the_plot["char_to_color_shape_updated"] = False  # used for meta
  this_game.the_plot["termination_countdown"] = None
  return this_game


def make_metalearning_game(
    object_properties, concept_type, explain="full",
    transformations_allowed=0, additional_extant_properties=None,
    agent_start=None, value_multiplier=1.,
    next_progressive_level=None, rng=None):
  """Constructs a metalearning version of the game.

  Args:
    object_properties: list of (character, position, position_type,
      shape, color, texture, value), for placing objects in the world.
    concept_type: one of ["color", "texture" "shape"], indicating
      which attribute has the odd-one-out.
    explain: One of "full" "reward", "properties" or "none." If none, no
      explanation. If "reward" the explanation describes whether the answer was
      correct or incorrect, and the features that show it. If "properties", will
      identify the properties of objects when adjacent to them. If "full", gives
      both properties + reward.
    transformations_allowed: number of transformations of object properties that
      the agent is allowed to make. Use 0 to match the original task, more to
      allow interesting interventions on the environment.
    additional_extant_properties: Optional dict, used to add properties that are
      desired but not yet present in the scene, for transformation. Should have
      as keys a subset of ["color", "texture", "shape"].
    agent_start: Optional agent start position (mostly for testing).
    value_multiplier: multiplies the rewards.
    next_progressive_level: if not None, the next level key to progress to.
    rng: An optional numpy Random Generator for choosing agent_start (if not
      set), to set a fixed seed use e.g. `rng=np.random.default_rng(seed=...)`

  Returns:
    this_game: Pycolab engine running the specified game.
  """
  if rng is None:
    rng = np.random.default_rng()

  char_to_color_shape = []
  drape_creators = {}
  forbidden_locations = []
  extant_attributes = {"color": set(),
                       "texture": set(),
                       "shape": set()}
  for obj in object_properties:
    # can't have player starting here
    forbidden_locations.append(obj.position)
    # explanations
    if explain not in ["none", "full", "reward", "properties"]:
      raise ValueError("Unrecognized explanation type: {}".format(explain))
    if explain in ["full", "properties"]:
      properties_string = "tbd"  # will be set later as needed.
    else:
      properties_string = ""
    explanation_string = ""
    if explain in ["full", "reward"]:
      explanation_string = "tbd"  # will be set later as needed.

    # create object builders
    drape_creators[obj.character] = ascii_art.Partial(
        MetaObjectDrape, object_properties=obj,
        properties_string=properties_string,
        explanation_string=explanation_string)
    char_to_color_shape.append(
        (obj.character, " ".join((obj.color, obj.texture, obj.shape))))
    extant_attributes["color"].add(obj.color)
    extant_attributes["texture"].add(obj.texture)
    extant_attributes["shape"].add(obj.shape)

  if additional_extant_properties is not None:
    for k, v in additional_extant_properties.items():
      extant_attributes[k].update(set(v))

  # set up agent start
  if agent_start is None:
    poss_starts = []
    for x in range(1, 10):
      for y in range(1, 10):
        if (x, y) not in forbidden_locations:
          poss_starts.append((x, y))
    agent_start = poss_starts[
        rng.integers(len(poss_starts))]
  sprites = {AGENT_CHAR: MetaPlayerSprite}
  # generate level and game
  level_layout = _generate_level_layout(object_properties, agent_start)
  this_game = ascii_art.ascii_art_to_game(
      art=level_layout,
      what_lies_beneath=" ",
      sprites=sprites,
      drapes=drape_creators,
      update_schedule=[[AGENT_CHAR],
                       [obj.character for obj in object_properties]])

  # update necessary plot information
  if transformations_allowed > 0:
    this_game.the_plot["instruction_string"] = "Make an odd one out"
  else:
    this_game.the_plot["instruction_string"] = "Find the odd one out"
  this_game.the_plot["explanation_string"] = ""
  this_game.the_plot["char_to_color_shape"] = dict(char_to_color_shape)
  this_game.the_plot["char_to_color_shape_updated"] = True
  this_game.the_plot["transformations_allowed"] = transformations_allowed
  this_game.the_plot["transformations_happening_now"] = False
  this_game.the_plot["extant_attributes"] = extant_attributes
  this_game.the_plot["concept_type"] = concept_type
  this_game.the_plot["termination_countdown"] = None
  this_game.the_plot["value_multiplier"] = value_multiplier
  this_game.the_plot["next_progressive_level"] = next_progressive_level
  this_game.the_plot["explain"] = explain
  return this_game


def main(argv):
  if len(argv) > 2:
    raise app.UsageError("Too many command-line arguments.")

  episode_type = argv[1]

  if episode_type == "basic":
    these_object_properties = [
        ObjectProperties(
            POSSIBLE_OBJECT_CHARS[1], (1, 4), "against_wall", "triangle", "red",
            "solid", INCORRECT_REWARD),
        ObjectProperties(
            POSSIBLE_OBJECT_CHARS[2], (4, 1), "against_wall", "triangle",
            "blue", "noise", INCORRECT_REWARD),
        ObjectProperties(
            POSSIBLE_OBJECT_CHARS[3], (4, 4), "in_center", "square", "red",
            "noise", CORRECT_REWARD),
        ObjectProperties(
            POSSIBLE_OBJECT_CHARS[4], (6, 6), "in_center", "triangle", "blue",
            "solid", INCORRECT_REWARD),
    ]
    game = make_game(object_properties=these_object_properties,
                     concept_type="shape")
  elif episode_type == "meta":

    these_object_properties = [
        ObjectProperties(
            POSSIBLE_OBJECT_CHARS[1], (1, 4), "against_wall", "triangle",
            "blue", "noise", INCORRECT_REWARD),
        ObjectProperties(
            POSSIBLE_OBJECT_CHARS[2], (4, 1), "against_wall", "triangle",
            "blue", "noise", INCORRECT_REWARD),
        ObjectProperties(
            POSSIBLE_OBJECT_CHARS[3], (4, 4), "in_center", "triangle", "blue",
            "noise", INCORRECT_REWARD),
        ObjectProperties(
            POSSIBLE_OBJECT_CHARS[4], (6, 6), "in_center", "triangle", "blue",
            "noise", INCORRECT_REWARD),
    ]

    game = make_metalearning_game(
        object_properties=these_object_properties,
        concept_type="shape", transformations_allowed=5, agent_start=(1, 6),
        additional_extant_properties={
            "color": ["red"],
            "shape": ["square"],
            "texture": ["solid"],
        })
  else:
    raise ValueError("Unrecognized argument: %s" % episode_type)

  # Note that these colors are only for human UI
  foreground_colors = {
      AGENT_CHAR: (999, 999, 999),  # Agent is white
      WALL_CHAR: (300, 300, 300),  # Wall, dark grey
      FLOOR_CHAR: (0, 0, 0),  # Floor
  }

  keys_to_actions = {
      # Basic movement.
      curses.KEY_UP: ACTIONS.MOVE_N,
      curses.KEY_DOWN: ACTIONS.MOVE_S,
      curses.KEY_LEFT: ACTIONS.MOVE_W,
      curses.KEY_RIGHT: ACTIONS.MOVE_E,
      -1: 11,  # Do nothing.
  }

  if episode_type == "basic":
    foreground_colors.update({
        POSSIBLE_OBJECT_CHARS[1]: (900, 100, 100),
        POSSIBLE_OBJECT_CHARS[2]: (100, 100, 900),
        POSSIBLE_OBJECT_CHARS[3]: (900, 100, 100),
        POSSIBLE_OBJECT_CHARS[4]: (100, 100, 900),
    })
  elif episode_type == "meta":
    keys_to_actions.update({
        "q": ACTIONS.TRANSFORM_COLOR,
        "w": ACTIONS.TRANSFORM_TEXTURE,
        "e": ACTIONS.TRANSFORM_SHAPE,
    })
    foreground_colors.update({
        POSSIBLE_OBJECT_CHARS[1]: (100, 100, 900),
        POSSIBLE_OBJECT_CHARS[2]: (100, 100, 900),
        POSSIBLE_OBJECT_CHARS[3]: (100, 100, 900),
        POSSIBLE_OBJECT_CHARS[4]: (100, 100, 900),
    })

  background_colors = {
      c: (0, 0, 0) for c in foreground_colors
  }
  ui = human_ui.CursesUi(
      keys_to_actions=keys_to_actions,
      delay=10000,
      colour_fg=foreground_colors,
      colour_bg=background_colors)

  ui.play(game)


if __name__ == "__main__":
  app.run(main)

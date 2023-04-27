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

"""Tests for the odd one out environment."""
from absl.testing import absltest
from absl.testing import parameterized

import dm_env

import numpy as np

from tell_me_why_explanations_rl import odd_one_out_environment
from tell_me_why_explanations_rl import odd_one_out_environment_core


class OddOneOutEnvironmentTest(parameterized.TestCase):

  @parameterized.parameters(
      "red solid triangle",
      "green horizontal_stripes plus",
      "blue vertical_stripes inverse_plus",
      "purple checker ex",
      "orange grid inverse_ex",
      "yellow noise circle",
      "brown solid tee",
      "pink horizontal_stripes upside_down_tee",
      "cyan vertical_stripes h",
      "dark_green checker u",
      "dark_red grid upside_down_u",
      "dark_blue noise square"
  )
  def test_template_gen(self, object_name):
    odd_one_out_environment._generate_template(object_name)

  def test_basic_core(self):
    chars = odd_one_out_environment_core.POSSIBLE_OBJECT_CHARS[:4]
    targ = chars[2]
    these_object_properties = [
        odd_one_out_environment_core.ObjectProperties(
            chars[0], (1, 4), "against_wall", "triangle", "red", "noise", 0.),
        odd_one_out_environment_core.ObjectProperties(
            chars[1], (4, 1), "against_wall", "triangle", "blue", "solid", 0.),
        odd_one_out_environment_core.ObjectProperties(
            targ, (4, 4), "in_center", "square", "red", "solid", 1.),
        odd_one_out_environment_core.ObjectProperties(
            chars[3], (6, 6), "in_center", "triangle", "blue", "noise", 0.),
    ]

    game = odd_one_out_environment_core.make_game(
        object_properties=these_object_properties,
        concept_type="shape", agent_start=(2, 2))
    obs, reward, discount = game.its_showtime()
    with self.subTest(name="ObjectAndAgentSpawns"):
      self.assertEqual(obs.board[4, 4], ord(targ))
      self.assertEqual(obs.board[2, 2],
                       ord(odd_one_out_environment_core.AGENT_CHAR))
      self.assertEqual(game.the_plot["char_to_color_shape"][targ],
                       "red solid square")
      self.assertEqual(game.the_plot["char_to_color_shape"][chars[0]],
                       "red noise triangle")
    with self.subTest(name="InitialObservationsAndInstructions"):
      self.assertIsNone(reward)
      self.assertEqual(discount, 1.)
      self.assertEqual(game.the_plot["instruction_string"],
                       "")
      self.assertEqual(game.the_plot["explanation_string"], "")
    with self.subTest(name="ActionAndPropertyDescriptions"):
      # step down and right
      obs, reward, _ = game.play(odd_one_out_environment_core.ACTIONS.MOVE_SE)
      self.assertIsNone(reward)
      self.assertEqual(obs.board[3, 3],
                       ord(odd_one_out_environment_core.AGENT_CHAR))
      self.assertEqual(game.the_plot["explanation_string"],
                       "This is a red solid square in_center")
    with self.subTest(name="GettingObjectAndExplanation"):
      # move to object
      obs, reward, _ = game.play(3)
      self.assertEqual(reward, 1.)
      self.assertEqual(game.the_plot["instruction_string"], "")
      self.assertEqual(game.the_plot["explanation_string"],
                       "Correct it is uniquely square")
    with self.subTest(name="ExplanationPhaseAndTermination"):
      # test termination of explanation phase
      self.assertFalse(game.game_over)
      game.play(odd_one_out_environment_core.ACTIONS.MOVE_W)
      game.play(odd_one_out_environment_core.ACTIONS.MOVE_W)
      game.play(odd_one_out_environment_core.ACTIONS.MOVE_E)
      self.assertFalse(game.game_over)
      game.play(odd_one_out_environment_core.ACTIONS.MOVE_E)
      self.assertTrue(game.game_over)

    with self.subTest(name="ChooseWrongObjectExplanations"):
      # play again but choose wrong object
      game = odd_one_out_environment_core.make_game(
          object_properties=these_object_properties,
          concept_type="shape", agent_start=(2, 2))
      _ = game.its_showtime()
      obs, reward, _ = game.play(odd_one_out_environment_core.ACTIONS.MOVE_SW)
      self.assertEqual(obs.board[3, 1],
                       ord(odd_one_out_environment_core.AGENT_CHAR))
      self.assertEqual(game.the_plot["explanation_string"],
                       "This is a blue solid triangle against_wall")
      obs, reward, _ = game.play(odd_one_out_environment_core.ACTIONS.MOVE_S)
      self.assertEqual(reward, 0.)
      self.assertEqual(
          game.the_plot["explanation_string"],
          "Incorrect other objects are blue solid triangle or against_wall")
      self.assertFalse(game.game_over)

  def test_confounding_core(self):
    chars = odd_one_out_environment_core.POSSIBLE_OBJECT_CHARS[:4]
    targ = chars[2]
    these_object_properties = [
        odd_one_out_environment_core.ObjectProperties(
            chars[0], (1, 4), "against_wall", "triangle", "blue", "solid", 0.),
        odd_one_out_environment_core.ObjectProperties(
            chars[1], (4, 1), "against_wall", "triangle", "blue", "solid", 0.),
        odd_one_out_environment_core.ObjectProperties(
            targ, (4, 4), "in_center", "square", "green", "noise", 1.),
        odd_one_out_environment_core.ObjectProperties(
            chars[3], (6, 6), "in_center", "triangle", "blue", "solid", 0.),
    ]
    game = odd_one_out_environment_core.make_game(
        object_properties=these_object_properties,
        concept_type="color", agent_start=(3, 3),
        explain_only_concept_type=True)
    obs, *_ = game.its_showtime()
    with self.subTest(name="AgentAndObjectSpawns"):
      self.assertEqual(obs.board[4, 4], ord(targ))
      self.assertEqual(obs.board[3, 3],
                       ord(odd_one_out_environment_core.AGENT_CHAR))
      self.assertEqual(game.the_plot["instruction_string"],
                       "")
    with self.subTest(name="PropertyExplanations"):
      self.assertEqual(game.the_plot["explanation_string"], "This is a green")
    with self.subTest(name="GettingObjectRewardExplanation"):
      # step down and right
      obs, reward, _ = game.play(odd_one_out_environment_core.ACTIONS.MOVE_SE)
      self.assertEqual(reward, 1.)
      self.assertEqual(game.the_plot["explanation_string"],
                       "Correct it is uniquely green")

  def test_meta_core(self):
    chars = odd_one_out_environment_core.POSSIBLE_OBJECT_CHARS[:4]
    targ = chars[0]
    these_object_properties = [
        odd_one_out_environment_core.ObjectProperties(
            targ, (1, 4), "against_wall", "triangle", "blue", "noise", 0.),
        odd_one_out_environment_core.ObjectProperties(
            chars[1], (4, 1), "against_wall", "triangle", "blue", "noise", 0.),
        odd_one_out_environment_core.ObjectProperties(
            chars[2], (4, 4), "in_center", "triangle", "blue", "noise", 0.),
        odd_one_out_environment_core.ObjectProperties(
            chars[3], (6, 6), "in_center", "triangle", "blue", "noise", 0.),
    ]

    game = odd_one_out_environment_core.make_metalearning_game(
        object_properties=these_object_properties,
        concept_type="shape", transformations_allowed=1, agent_start=(1, 6),
        additional_extant_properties={
            "color": ["red"],
            "shape": ["square"],
            "texture": ["solid"],
        })
    obs, reward, discount = game.its_showtime()
    with self.subTest(name="InitialSpawnsObservationsAndInstructions"):
      self.assertEqual(obs.board[1, 4], ord(targ))
      self.assertEqual(obs.board[1, 6],
                       ord(odd_one_out_environment_core.AGENT_CHAR))
      self.assertIsNone(reward)
      self.assertEqual(discount, 1.)
      self.assertEqual(game.the_plot["char_to_color_shape"][targ],
                       "blue noise triangle")
      self.assertEqual(game.the_plot["instruction_string"],
                       "Make an odd one out")
      self.assertEqual(game.the_plot["explanation_string"], "")
    with self.subTest(name="PropertyExplanations"):
      # step left
      obs, reward, _ = game.play(odd_one_out_environment_core.ACTIONS.MOVE_W)
      self.assertIsNone(reward)
      self.assertEqual(obs.board[1, 5],
                       ord(odd_one_out_environment_core.AGENT_CHAR))
      self.assertEqual(game.the_plot["instruction_string"],
                       "Make an odd one out")
      self.assertEqual(game.the_plot["explanation_string"],
                       "This is a blue noise triangle")
      self.assertFalse(game.the_plot["transformations_happening_now"])
    with self.subTest(name="TransformShape"):
      # transform shape
      obs, reward, _ = game.play(
          odd_one_out_environment_core.ACTIONS.TRANSFORM_SHAPE)
      self.assertIsNone(reward)
      self.assertEqual(obs.board[1, 5],
                       ord(odd_one_out_environment_core.AGENT_CHAR))
      self.assertTrue(game.the_plot["transformations_happening_now"])
      self.assertTrue(game.the_plot["char_to_color_shape_updated"])
      self.assertEqual(game.the_plot["char_to_color_shape"][targ],
                       "blue noise square")
      self.assertEqual(game.the_plot["instruction_string"],
                       "Make an odd one out")
      self.assertEqual(game.the_plot["explanation_string"],
                       "This is a blue noise square")
    with self.subTest(name="GetObjectAndRewardExplanation"):
      # move to transformed object
      obs, reward, _ = game.play(odd_one_out_environment_core.ACTIONS.MOVE_W)
      self.assertEqual(reward, 1.)
      self.assertFalse(game.the_plot["transformations_happening_now"])
      self.assertEqual(game.the_plot["instruction_string"], "")
      self.assertEqual(game.the_plot["explanation_string"],
                       "Correct the concept is shape and it is uniquely square")

    with self.subTest(name="FailWithoutTransformations"):
      # now play same game without a transformation allowed (impossible to win)
      game = odd_one_out_environment_core.make_metalearning_game(
          object_properties=these_object_properties,
          concept_type="shape", transformations_allowed=0, agent_start=(1, 6),
          additional_extant_properties={
              "color": ["red"],
              "shape": ["square"],
              "texture": ["solid"],
          })
      obs, reward, discount = game.its_showtime()
      self.assertEqual(game.the_plot["instruction_string"],
                       "Find the odd one out")
      self.assertEqual(game.the_plot["explanation_string"], "")
      # step left
      obs, reward, _ = game.play(odd_one_out_environment_core.ACTIONS.MOVE_W)
      # fail to transform shape, not allowed
      obs, reward, _ = game.play(
          odd_one_out_environment_core.ACTIONS.TRANSFORM_SHAPE)
      self.assertIsNone(reward)
      self.assertEqual(obs.board[1, 5],
                       ord(odd_one_out_environment_core.AGENT_CHAR))
      self.assertFalse(game.the_plot["transformations_happening_now"])
      self.assertEqual(game.the_plot["char_to_color_shape"][targ],
                       "blue noise triangle")
      self.assertEqual(game.the_plot["explanation_string"],
                       "This is a blue noise triangle")
      # move to object, no reward
      obs, reward, _ = game.play(odd_one_out_environment_core.ACTIONS.MOVE_W)
      self.assertEqual(reward, 0.)
      self.assertFalse(game.the_plot["transformations_happening_now"])
      self.assertEqual(game.the_plot["instruction_string"], "")
      self.assertEqual(
          game.the_plot["explanation_string"],
          "Incorrect the concept is shape and other objects are triangle")

  def test_full_base_game(self):
    def _game_factory():
      chars = odd_one_out_environment_core.POSSIBLE_OBJECT_CHARS[:4]
      these_object_properties = [
          odd_one_out_environment_core.ObjectProperties(
              chars[0], (1, 4), "against_wall", "triangle", "red", "noise", 0.),
          odd_one_out_environment_core.ObjectProperties(
              chars[1], (4, 1), "against_wall", "triangle", "blue", "solid",
              0.),
          odd_one_out_environment_core.ObjectProperties(
              chars[2], (4, 4), "in_center", "square", "red", "solid", 1.),
          odd_one_out_environment_core.ObjectProperties(
              chars[3], (6, 6), "in_center", "triangle", "blue", "noise", 0.),
      ]

      return odd_one_out_environment_core.make_game(
          object_properties=these_object_properties,
          concept_type="shape", agent_start=(2, 2))
    env = odd_one_out_environment.OddOneOutEnvironment(
        game_factory=_game_factory, rng=np.random.default_rng(seed=1234))
    result = env.reset()
    with self.subTest(name="InitialObservations"):
      view_size = odd_one_out_environment.SCROLL_CROP_SIZE
      upsample_size = odd_one_out_environment.UPSAMPLE_SIZE
      self.assertLen(result.observation, 2)
      self.assertEqual(result.observation[0].shape,
                       (view_size * upsample_size,
                        view_size * upsample_size,
                        3))
      offset = upsample_size * (view_size // 2)
      # check egocentric scrolling is working, by checking agent is in center
      agent_template = odd_one_out_environment._CHAR_TO_TEMPLATE_BASE[
          odd_one_out_environment_core.AGENT_CHAR]
      np.testing.assert_array_almost_equal(
          result.observation[0][offset:offset + upsample_size,
                                offset:offset + upsample_size, :],
          agent_template / 255.)
      # no explanation yet
      self.assertEqual(result.observation[1].item(),
                       "")
    with self.subTest(name="StepAndObserve"):
      # step down and right
      result = env.step(odd_one_out_environment_core.ACTIONS.MOVE_SE)
      np.testing.assert_array_almost_equal(
          result.observation[0][offset:offset + upsample_size,
                                offset:offset + upsample_size, :],
          agent_template / 255.)
      self.assertEqual(result.reward, 0.)
      self.assertEqual(result.observation[1].item(),
                       "This is a red solid square in_center")

  def test_single_meta_game(self):
    def _game_factory(tr_al=1):
      chars = odd_one_out_environment_core.POSSIBLE_OBJECT_CHARS[:4]
      these_object_properties = [
          odd_one_out_environment_core.ObjectProperties(
              chars[0], (1, 4), "against_wall", "triangle", "blue", "noise",
              0.),
          odd_one_out_environment_core.ObjectProperties(
              chars[1], (4, 1), "against_wall", "triangle", "blue", "noise",
              0.),
          odd_one_out_environment_core.ObjectProperties(
              chars[2], (4, 4), "in_center", "triangle", "blue", "noise", 0.),
          odd_one_out_environment_core.ObjectProperties(
              chars[3], (6, 6), "in_center", "triangle", "blue", "noise", 0.),
      ]

      return odd_one_out_environment_core.make_metalearning_game(
          object_properties=these_object_properties,
          concept_type="shape", transformations_allowed=tr_al,
          agent_start=(1, 6),
          additional_extant_properties={
              "color": ["red"],
              "shape": ["square"],
              "texture": ["solid"],
          })
    env = odd_one_out_environment.MetaOddOneOutEnvironment(
        game_factory=_game_factory, rng=np.random.default_rng(seed=1234))
    result = env.reset()
    with self.subTest(name="InitialObservations"):
      self.assertLen(result.observation, 3)
      view_size = odd_one_out_environment.SCROLL_CROP_SIZE
      upsample_size = odd_one_out_environment.UPSAMPLE_SIZE
      self.assertEqual(result.observation[0].shape,
                       (view_size * upsample_size,
                        view_size * upsample_size,
                        3))
      self.assertEqual(result.observation[1].item(), "")
      self.assertEqual(result.observation[2].item(), "Make an odd one out")
    with self.subTest(name="StepAndPropertyExplanation"):
      result = env.step(odd_one_out_environment_core.ACTIONS.MOVE_W)
      self.assertEqual(result.observation[1].item(),
                       "This is a blue noise triangle")
      self.assertEqual(result.observation[2].item(), "Make an odd one out")
    with self.subTest(name="InstructionChangesWithNoTransforms"):
      # now try with no transforms, to check instruction changes
      env = odd_one_out_environment.MetaOddOneOutEnvironment(
          game_factory=lambda: _game_factory(tr_al=0),
          rng=np.random.default_rng(seed=1234))
      result = env.reset()
      self.assertLen(result.observation, 3)
      self.assertEqual(result.observation[2].item(), "Find the odd one out")

  def test_meta_full_task(self):
    env = odd_one_out_environment.metalearning_builder(
        num_trials_before_test=1, intervention_type="easy",
        concept_type="shape", explain="full", rng=np.random.default_rng(seed=1))
    result = env.reset()

    with self.subTest(name="InitialObservations"):
      self.assertLen(result.observation, 3)
      self.assertEqual(result.observation[1].item(), "")
      self.assertEqual(result.observation[2].item(), "Make an odd one out")
    with self.subTest(name="StepAndObserve"):
      result = env.step(odd_one_out_environment_core.ACTIONS.MOVE_W)
      self.assertEqual(result.observation[1].item(), "")
      result = env.step(odd_one_out_environment_core.ACTIONS.MOVE_W)
      self.assertEqual(result.observation[1].item(),
                       "This is a peach noise plus")
      self.assertEqual(result.reward, 0.)
    with self.subTest(name="ObjectChangesWithTransformation"):
      # check object rendering changes before -> after transformation
      scs = odd_one_out_environment.SCROLL_CROP_SIZE
      us = odd_one_out_environment.UPSAMPLE_SIZE
      off = us * (scs // 2)
      pre_object_image = result.observation[0][off:off + us, off - us:off, :]
      result = env.step(odd_one_out_environment_core.ACTIONS.TRANSFORM_SHAPE)
      self.assertEqual(result.observation[1].item(),
                       "This is a peach noise u")
      self.assertEqual(result.reward, 0.)
      post_object_image = result.observation[0][off:off + us, off - us:off, :]
      self.assertFalse(np.array_equal(pre_object_image, post_object_image))
    with self.subTest(name="GetObjectAndRewardExplanation"):
      # now grab
      result = env.step(odd_one_out_environment_core.ACTIONS.MOVE_W)
      self.assertEqual(result.observation[1].item(),
                       "Correct the concept is shape and it is uniquely u")
      self.assertEqual(result.reward, 1.)
      result = env.step(odd_one_out_environment_core.ACTIONS.MOVE_E)
      result = env.step(odd_one_out_environment_core.ACTIONS.MOVE_E)
      result = env.step(odd_one_out_environment_core.ACTIONS.MOVE_W)
      self.assertEqual(result.observation[1].item(),
                       "Correct the concept is shape and it is uniquely u")
      self.assertEqual(result.reward, 0.)
    with self.subTest(name="CheckTrialTransition"):
      # should trigger level transition to test level
      result = env.step(odd_one_out_environment_core.ACTIONS.MOVE_W)
      self.assertEqual(result.observation[1].item(), "")
      self.assertEqual(result.observation[2].item(), "Find the odd one out")
      self.assertEqual(result.reward, 0.)
      self.assertEqual(result.step_type, dm_env.StepType.MID)
    with self.subTest(name="FinalTrial"):
      result = env.step(odd_one_out_environment_core.ACTIONS.MOVE_NE)
      result = env.step(odd_one_out_environment_core.ACTIONS.MOVE_NE)
      self.assertEqual(result.observation[1].item(),
                       "This is a lavender checker triangle")
      result = env.step(odd_one_out_environment_core.ACTIONS.MOVE_NE)
      self.assertEqual(result.reward, 10.)
      self.assertEqual(
          result.observation[1].item(),
          "Correct the concept is shape and it is uniquely triangle")

  def test_confound_builder(self):
    env = odd_one_out_environment.confounding_builder(
        confounding="confounded", concept_type="texture", explain="full",
        rng=np.random.default_rng(seed=1))
    result = env.reset()
    self.assertLen(result.observation, 2)
    current_objects = env._current_game.the_plot["char_to_color_shape"].values()
    self.assertLen(set(current_objects), 2)
    # deconfounded version
    env = odd_one_out_environment.confounding_builder(
        confounding="deconfounded", concept_type="texture", explain="full",
        rng=np.random.default_rng(seed=1))
    result = env.reset()
    current_objects = env._current_game.the_plot["char_to_color_shape"].values()
    self.assertLen(set(current_objects), 4)

  @parameterized.parameters(
      "position_none",
      "shape_full",
      "texture_reward",
      "color_properties",
      ### confounding levels
      "confounding_color_none",
      "confounding_shape_full",
      "deconfounding_texture_full",
      ### metalearning levels
      "meta_3_easy_shape_full",
      "meta_3_hard1_color_none",
      "meta_3_hard3_texture_full",
  )
  def test_from_name(self, level_name):
    np.random.seed(0)
    env = odd_one_out_environment.from_name(level_name)
    env.reset()
    view_size = odd_one_out_environment.SCROLL_CROP_SIZE
    upsample_size = odd_one_out_environment.UPSAMPLE_SIZE
    for i in range(8):
      result = env.step(i).observation
      self.assertEqual(result[0].shape,
                       (view_size * upsample_size,
                        view_size * upsample_size,
                        3))

if __name__ == "__main__":
  absltest.main()

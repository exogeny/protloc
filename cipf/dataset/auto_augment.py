from typing import Any, Optional, Iterable, Tuple, Text

import inspect
import numpy as np
import tensorflow as tf

from cipf.dataset.augment import parse_policy_info


def _apply_func_with_prob(func: Any,
                          image: tf.Tensor,
                          bboxes: Optional[tf.Tensor],
                          arguments: Any,
                          prob: float):
  """Apply `func` to image with probability `prob`."""
  assert isinstance(arguments, tuple)
  assert inspect.getfullargspec(func)[0][1] == 'bboxes'

  should_apply_op = tf.cast(
      tf.floor(tf.random.uniform([], dtype=tf.float32) + prob), tf.bool)
  augmented_image, augmented_bboxes = tf.cond(
      should_apply_op,
      lambda: func(image, bboxes, *arguments),
      lambda: (image, bboxes))
  return augmented_image, augmented_bboxes


def _select_and_apply_random_policy(policies: Any,
                                    image: tf.Tensor,
                                    bboxes: Optional[tf.Tensor] = None):
  """Select a random policy from `policies` and apply it to `image`."""
  policy_to_select = tf.random.uniform([], maxval=len(policies), dtype=tf.int32)
  # Note that using tf.case instead of tf.conds would result in significantly
  # larger graphs and would even break export for some larger policies.
  for (i, policy) in enumerate(policies):
    image, bboxes = tf.cond(
        tf.equal(i, policy_to_select),
        lambda selected_policy=policy: selected_policy(image, bboxes),
        lambda: (image, bboxes))
  return image, bboxes



class AutoAugment:
  """Applies the AutoAugment policy to images.

    AutoAugment is from the paper: https://arxiv.org/abs/1805.09501.
  """

  def __init__(
      self,
      augmentation_name: Text = 'v0',
      cutout_const: float = 100,
      translate_const: float = 250):
    self._augmentation_name = augmentation_name
    self._cutout_const = cutout_const
    self._translate_const = translate_const
    self._available_policies = {
        'v0': self.policy_v0(),
        'test': self.policy_test(),
        'simple': self.policy_simple(),
        'vit': self.vit(),
        'protein': self.protein(),
    }

    # Not using the custom policy for now
    if augmentation_name not in self._available_policies:
      raise ValueError(f'Unknown augmentation policy: {augmentation_name}')
    self._policies = self._available_policies[augmentation_name]

  def _check_policy(self, policies):
    in_shape = np.array(policies).shape
    if len(in_shape) != 3 or (in_shape[-1:] != (3,) and in_shape[-1:] != (1,)):
      raise ValueError('Wrong shape detected for custom policy. Expected '
                       '(:, :, 3) or (:, :, 1) but got {}.'.format(in_shape))

  def _make_tf_policies(self):
    replace_value = [128] * 3
    # func is the string name of the augmentation function, prob is the
    # probability of applying the operation and level is the parameter
    # associated with the tf op.

    # tf_policies are functions that take in an image and return an augmented
    # image.
    tf_policies = []
    for policy in self._policies:
      tf_policy = []
      assert_ranges = []

      # Link string name to the correct python function and make sure the
      # correct argument is passed into that function.
      for policy_info in policy:
        _, prob, level = policy_info
        assert_ranges.append(tf.Assert(tf.less_equal(prob, 1.), [prob]))
        assert_ranges.append(tf.Assert(tf.less_equal(level, 10), [level]))

        policy_info = list(policy_info) + [
          replace_value, self._cutout_const, self._translate_const
        ]
        tf_policy.append(parse_policy_info(*policy_info))

      # Now build the tf policy that will apply the augmentation procedue
      # on image
      def make_final_policy(tf_policy_):
        def final_policy(_image, _bboxes):
          for func, prob, arguments in tf_policy_:
            _image, _bboxes = _apply_func_with_prob(
                func, _image, _bboxes, arguments, prob)
          return _image, _bboxes
        return final_policy

      with tf.control_dependencies(assert_ranges):
        tf_policies.append(make_final_policy(tf_policy))
    return tf_policies

  def distort(self,
              image: tf.Tensor,
              bbxoes: Optional[tf.Tensor] = None) -> tf.Tensor:
    input_image_dtype = image.dtype
    if input_image_dtype != tf.uint8:
      image = tf.image.convert_image_dtype(image, dtype=tf.uint8)

    tf_policies = self._make_tf_policies()
    image, _ = _select_and_apply_random_policy(tf_policies, image, bbxoes)
    image = tf.cast(image, dtype=input_image_dtype)
    return image

  @staticmethod
  def policy_v0():
    return [
        [('Equalize', 0.8, 1), ('ShearY', 0.8, 4)],
        [('Color', 0.4, 9), ('Equalize', 0.6, 3)],
        [('Color', 0.4, 1), ('Rotate', 0.6, 8)],
        [('Solarize', 0.8, 3), ('Equalize', 0.4, 7)],
        [('Solarize', 0.4, 2), ('Solarize', 0.6, 2)],
        [('Color', 0.2, 0), ('Equalize', 0.8, 8)],
        [('Equalize', 0.4, 8), ('SolarizeAdd', 0.8, 3)],
        [('ShearX', 0.2, 9), ('Rotate', 0.6, 8)],
        [('Color', 0.6, 1), ('Equalize', 1.0, 2)],
        [('Invert', 0.4, 9), ('Rotate', 0.6, 0)],
        [('Equalize', 1.0, 9), ('ShearY', 0.6, 3)],
        [('Color', 0.4, 7), ('Equalize', 0.6, 0)],
        [('Posterize', 0.4, 6), ('AutoContrast', 0.4, 7)],
        [('Solarize', 0.6, 8), ('Color', 0.6, 9)],
        [('Solarize', 0.2, 4), ('Rotate', 0.8, 9)],
        [('Rotate', 1.0, 7), ('TranslateY', 0.8, 9)],
        [('ShearX', 0.0, 0), ('Solarize', 0.8, 4)],
        [('ShearY', 0.8, 0), ('Color', 0.6, 4)],
        [('Color', 1.0, 0), ('Rotate', 0.6, 2)],
        [('Equalize', 0.8, 4), ('Equalize', 0.0, 8)],
        [('Equalize', 1.0, 4), ('AutoContrast', 0.6, 2)],
        [('ShearY', 0.4, 7), ('SolarizeAdd', 0.6, 7)],
        [('Posterize', 0.8, 2), ('Solarize', 0.6, 10)],
        [('Solarize', 0.6, 8), ('Equalize', 0.6, 1)],
        [('Color', 0.8, 6), ('Rotate', 0.4, 5)],
    ]

  @staticmethod
  def policy_test():
    return [
        [('TranslateX', 1.0, 4), ('Equalize', 1.0, 10)],
    ]

  @staticmethod
  def policy_simple():
    return [
        [('Color', 0.4, 9), ('Equalize', 0.6, 3)],
        [('Solarize', 0.8, 3), ('Equalize', 0.4, 7)],
        [('Solarize', 0.4, 2), ('Solarize', 0.6, 2)],
        [('Color', 0.2, 0), ('Equalize', 0.8, 8)],
        [('Equalize', 0.4, 8), ('SolarizeAdd', 0.8, 3)],
        [('Color', 0.6, 1), ('Equalize', 1.0, 2)],
        [('Color', 0.4, 7), ('Equalize', 0.6, 0)],
        [('Posterize', 0.4, 6), ('AutoContrast', 0.4, 7)],
        [('Solarize', 0.6, 8), ('Color', 0.6, 9)],
        [('Equalize', 0.8, 4), ('Equalize', 0.0, 8)],
        [('Equalize', 1.0, 4), ('AutoContrast', 0.6, 2)],
        [('Posterize', 0.8, 2), ('Solarize', 0.6, 10)],
        [('Solarize', 0.6, 8), ('Equalize', 0.6, 1)],
    ]

  @staticmethod
  def vit():
    return [
        [('Sharpness', 0.4, 1.4), ('Brightness', 0.2, 2.0), ('Cutout', 0.8, 8)],
        [('Equalize', 0.0, 1.8), ('Contrast', 0.2, 2.0), ('Cutout', 0.8, 8)],
        [('Sharpness', 0.2, 1.8), ('Color', 0.2, 1.8), ('Cutout', 0.8, 8)],
        [('Solarize', 0.2, 1.4), ('Equalize', 0.6, 1.8), ('Cutout', 0.8, 8)],
        [('Sharpness', 0.2, 0.2), ('Equalize', 0.2, 1.4), ('Cutout', 0.8, 8)],
        [('Sharpness', 0.4, 7), ('Invert', 0.6, 8), ('Cutout', 0.8, 8)],
        [('Invert', 0.6, 4), ('Equalize', 1.0, 8), ('Cutout', 0.8, 8)],
        [('Posterize', 0.6, 7), ('Posterize', 0.6, 6), ('Cutout', 0.8, 8)],
        [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5), ('Cutout', 0.8, 8)],
    ]

  @staticmethod
  def protein():
    return [
        [('TranslateX', 0.7, 5), ('ShearY', 0.2, 7)],
        [('TranslateY', 0.7, 5), ('ShearX', 0.2, 7)],
        [('Rotate', 0.6, 8), ('Sharpness', 0.2, 1.8)],
        [('AutoContrast', 0.6, 5), ('Cutout', 0.8, 8)],
        [('Equalize', 0.6, 1.8), ('Cutout', 0.8, 8)],
        [('Contrast', 0.2, 2.0), ('Cutout', 0.8, 8)],
        [('Brightness', 0.2, 2.0), ('Cutout', 0.8, 8)],
    ]

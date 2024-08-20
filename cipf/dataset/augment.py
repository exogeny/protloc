from typing import Any, List, Tuple, Union, Optional
import math
import inspect
import numpy as np
import tensorflow as tf


# This signifies the max integer that the controller RNN could predict for the
# augmentation scheme.
_MAX_LEVEL = 10.

# Functions that have a 'prob' parameter
PROB_FUNCS = frozenset({
    'TranslateY_Only_BBoxes',
})

# Represents an invalid bounding box that is used for checking for padding
# lists of bounding box coordinates for a few augmentation operations
_INVALID_BOX = [[-1.0, -1.0, -1.0, -1.0]]

# Functions that have a 'replace' parameter
REPLACE_FUNCS = frozenset({
    'Rotate',
    'TranslateX',
    'ShearX',
    'ShearY',
    'TranslateY',
    'Cutout',
    'Rotate_BBox',
    'ShearX_BBox',
    'ShearY_BBox',
    'TranslateX_BBox',
    'TranslateY_BBox',
    'TranslateY_Only_BBoxes',
})


def _fill_rectangle(image: tf.Tensor,
                    center_width: int,
                    center_height: int,
                    half_width: int,
                    half_height: int,
                    replace: Optional[int] = None) -> tf.Tensor:
  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]
  lower_pad = tf.maximum(0, center_height - half_height)
  upper_pad = tf.maximum(0, image_height - center_height - half_height)
  left_pad = tf.maximum(0, center_width - half_width)
  right_pad = tf.maximum(0, image_width - center_width - half_width)
  cutout_shape = [
      image_height - (lower_pad + upper_pad),
      image_width - (left_pad + right_pad)
  ]
  padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
  mask = tf.pad(
      tf.zeros(cutout_shape, dtype=image.dtype),
      padding_dims,
      constant_values=1)
  mask = tf.expand_dims(mask, -1)
  mask = tf.tile(mask, [1, 1, 3])

  if replace is None:
    fill = tf.random.normal(tf.shape(image), dtype=image.dtype)
  elif isinstance(replace, tf.Tensor):
    fill = replace
  else:
    fill = tf.ones_like(image, dtype=image.dtype) * replace
  image = tf.where(tf.equal(mask, 0), fill, image)
  return image


def _clip_bbox(min_y, min_x, max_y, max_x):
  """Clip bounding box coordinates between 0 and 1."""
  min_y = tf.clip_by_value(min_y, 0.0, 1.0)
  min_x = tf.clip_by_value(min_x, 0.0, 1.0)
  max_y = tf.clip_by_value(max_y, 0.0, 1.0)
  max_x = tf.clip_by_value(max_x, 0.0, 1.0)
  return min_y, min_x, max_y, max_x


def _check_bbox_area(min_y, min_x, max_y, max_x, delta: float = 0.05):
  height = max_y - min_y
  width = max_x - min_x
  def _adjust_bbox_boundaries(min_coord, max_coord):
    # Make sure max is never 0 and min is never 1
    max_coord = tf.maximum(max_coord, 0.0 + delta)
    min_coord = tf.minimum(min_coord, 1.0 - delta)
    return min_coord, max_coord
  min_y, max_y = tf.cond(tf.equal(height, 0.0),
                         lambda: _adjust_bbox_boundaries(min_y, max_y),
                         lambda: (min_y, max_y))
  min_x, max_x = tf.cond(tf.equal(width, 0.0),
                         lambda: _adjust_bbox_boundaries(min_x, max_x),
                         lambda: (min_x, max_x))
  return min_y, min_x, max_y, max_x


def _rotate_bbox(bbox, image_height, image_width, degrees):
  """Rotates the bbox coordinated by degrees."""
  image_height = tf.cast(image_height, tf.float32)
  image_width = tf.cast(image_width, tf.float32)
  # Convert from degrees to radians.
  degrees_to_radians = math.pi / 180.0
  radians = degrees * degrees_to_radians
  # Translate the bbox to the center of the image and turn the normalized 0-1
  # coordinates to absolute pixel locations.
  # Y coordinates are made negative as the y axis of images goes down with
  # increasing pixel values, so we negate to make sure x axis and y axis points
  # are in the traditionally positive direction.
  min_y = -tf.cast(image_height * (bbox[0] - 0.5), tf.float32)
  min_x = tf.cast(image_width * (bbox[1] - 0.5), tf.float32)
  max_y = -tf.cast(image_height * (bbox[2] - 0.5), tf.float32)
  max_x = tf.cast(image_width * (bbox[3] - 0.5), tf.float32)
  coordinates = tf.stack(
      [[min_y, min_x], [min_y, max_x], [max_y, min_x], [max_y, max_x]])
  coordinates = tf.cast(coordinates, tf.float32)
  # Rotate the coordinates according to the rotation matrix clockwise if
  # radians is positive, else negative
  rotation_matrix = tf.stack([
      [tf.cos(radians), tf.sin(radians)],
      [-tf.sin(radians), tf.cos(radians)]
  ])
  new_coords = tf.cast(
      tf.matmul(rotation_matrix, tf.transpose(coordinates)), tf.int32)
  # Find min/max values and convert them back to normalized 0-1 coordinates.
  min_y = -(
      tf.cast(tf.reduce_max(new_coords[0, :]), tf.float32) / image_height - 0.5)
  min_x = (
      tf.cast(tf.reduce_min(new_coords[1, :]), tf.float32) / image_width + 0.5)
  max_y = -(
      tf.cast(tf.reduce_min(new_coords[0, :]), tf.float32) / image_height - 0.5)
  max_x = (
      tf.cast(tf.reduce_max(new_coords[1, :]), tf.float32) / image_width + 0.5)
  # Clip the bboxes to be sure the fall between [0, 1].
  min_y, min_x, max_y, max_x = _clip_bbox(min_y, min_x, max_y, max_x)
  min_y, min_x, max_y, max_x = _check_bbox_area(min_y, min_x, max_y, max_x)
  return tf.stack([min_y, min_x, max_y, max_x])


def _shift_bbox(bbox, image_height, image_width, pixels, shift_horizontal):
  pixels = tf.cast(pixels, tf.float32)
  # Convert bbox to integer pioxel locations.
  min_y = tf.cast(tf.cast(image_height, tf.float32) * bbox[0], tf.float32)
  min_x = tf.cast(tf.cast(image_width, tf.float32) * bbox[1], tf.float32)
  max_y = tf.cast(tf.cast(image_height, tf.float32) * bbox[2], tf.float32)
  max_x = tf.cast(tf.cast(image_width, tf.float32) * bbox[3], tf.float32)
  if shift_horizontal:
    min_x = tf.maximum(0.0, min_x - pixels)
    max_x = tf.minimum(image_width, max_x - pixels)
  else:
    min_y = tf.maximum(0.0, min_y - pixels)
    max_y = tf.minimum(image_height, max_y - pixels)
  # Convert bbox back to floats.
  min_y = tf.cast(min_y, tf.float32) / tf.cast(image_height, tf.float32)
  min_x = tf.cast(min_x, tf.float32) / tf.cast(image_width, tf.float32)
  max_y = tf.cast(max_y, tf.float32) / tf.cast(image_height, tf.float32)
  max_x = tf.cast(max_x, tf.float32) / tf.cast(image_width, tf.float32)
  # Clip the bboxes to be sure the fall between [0, 1].
  min_y, min_x, max_y, max_x = _clip_bbox(min_y, min_x, max_y, max_x)
  min_y, min_x, max_y, max_x = _check_bbox_area(min_y, min_x, max_y, max_x)
  return tf.stack([min_y, min_x, max_y, max_x])


def _shear_bbox(bbox, image_height, image_width, level, shear_horizontal):
  image_height = tf.cast(image_height, tf.float32)
  image_width = tf.cast(image_width, tf.float32)
  # Change bbox coordinates to be pixels
  min_y = tf.cast(image_height * bbox[0], tf.float32)
  min_x = tf.cast(image_width * bbox[1], tf.float32)
  max_y = tf.cast(image_height * bbox[2], tf.float32)
  max_x = tf.cast(image_width * bbox[3], tf.float32)
  coordinates = tf.stack(
    [
      [min_y, min_x],
      [min_y, max_x],
      [max_y, min_x],
      [max_y, max_x],
    ]
  )
  coordinates = tf.cast(coordinates, tf.float32)
  # Shear the coordinates according to the translation matrix
  if shear_horizontal:
    translation_matrix = tf.stack(
      [
        [1, 0],
        [-level, 1],
      ]
    )
  else:
    translation_matrix = tf.stack(
      [
        [1, -level],
        [0, 1],
      ]
    )
  translation_matrix = tf.cast(translation_matrix, tf.float32)
  new_coords = tf.cast(
    tf.matmul(translation_matrix, tf.transpose(coordinates)), tf.int32)
  # Find min/max values and convert them back to float
  min_y = tf.cast(tf.reduce_min(new_coords[0, :]), tf.float32) / image_height
  min_x = tf.cast(tf.reduce_min(new_coords[1, :]), tf.float32) / image_width
  max_y = tf.cast(tf.reduce_max(new_coords[0, :]), tf.float32) / image_height
  max_x = tf.cast(tf.reduce_max(new_coords[1, :]), tf.float32) / image_width
  # Clip the bboxes to be sure the fall between [0, 1].
  min_y, min_x, max_y, max_x = _clip_bbox(min_y, min_x, max_y, max_x)
  min_y, min_x, max_y, max_x = _check_bbox_area(min_y, min_x, max_y, max_x)
  return tf.stack([min_y, min_x, max_y, max_x])


def _concat_bbox(bbox, bboxes):
  # Note if all elements in bboxes are -1 (_INVALID_BOX), then this means
  # we discard bboxes and start the bboxes Tensor with the current bbox.
  bboxes_sum_check = tf.reduce_sum(bboxes)
  bbox = tf.expand_dims(bbox, 0)
  # This check will be true when it is an _INVALID_BOX
  bboxes = tf.cond(tf.equal(bboxes_sum_check, -4.0),
                   lambda: bbox,
                   lambda: tf.concat([bboxes, bbox], 0))
  return bboxes


def _apply_bbox_augmentation(image,
                             bbox,
                             augmentation_func,
                             *args):
  image_height = tf.cast(tf.shape(image)[0], tf.float32)
  image_width = tf.cast(tf.shape(image)[1], tf.float32)
  min_y = tf.cast(image_height * bbox[0], tf.int32)
  min_x = tf.cast(image_width * bbox[1], tf.int32)
  max_y = tf.cast(image_height * bbox[2], tf.int32)
  max_x = tf.cast(image_width * bbox[3], tf.int32)
  image_height = tf.cast(image_height, tf.int32)
  image_width = tf.cast(image_width, tf.int32)

  # Clip to be sure the max values do not fall out of range.
  max_y = tf.minimum(max_y, image_height - 1)
  max_x = tf.minimum(max_x, image_width - 1)

  # Get the sub-tensor that is the image within the bounding box region.
  bbox_content = image[min_y:max_y + 1, min_x:max_x + 1, :]

  # Apply the augmentation function to the bbox portion of the image.
  augmented_bbox_content = augmentation_func(bbox_content, *args)

  # Pad the augmented_bbox_content and the mask to match the shape of original
  # image.
  augmented_bbox_content = tf.pad(augmented_bbox_content,
                                  [[min_y, (image_height - 1) - max_y],
                                   [min_x, (image_width - 1) - max_x],
                                   [0, 0]])

  # Create a mask that will be used to zero out a part of the original image.
  mask_tensor = tf.zeros_like(bbox_content)

  mask_tensor = tf.pad(mask_tensor,
                       [[min_y, (image_height - 1) - max_y],
                        [min_x, (image_width - 1) - max_x],
                        [0, 0]],
                       constant_values=1)
  # Replace the old bbox content with the new augmented content.
  image = image * mask_tensor + augmented_bbox_content
  return image


def _apply_bbox_augmentation_wrapper(image,
                                     bbox,
                                     new_bboxes,
                                     prob,
                                     augmentation_func,
                                     func_changes_bbox,
                                     *args):
  should_apply_op = tf.cast(
      tf.floor(tf.random.uniform([], dtype=tf.float32) + prob), tf.bool)
  if func_changes_bbox:
    augmented_image, bbox = tf.cond(
        should_apply_op,
        lambda: augmentation_func(image, bbox, *args),
        lambda: (image, bbox))
  else:
    augmented_image = tf.cond(
        should_apply_op,
        lambda: _apply_bbox_augmentation(image, bbox, augmentation_func, *args),
        lambda: image)
  new_bboxes = _concat_bbox(bbox, new_bboxes)
  return augmented_image, new_bboxes


def _apply_multi_bbox_augmentation(image,
                                   bboxes,
                                   prob,
                                   aug_func,
                                   func_changes_bbox,
                                   *args):
  if image.shape.rank == 4:
    raise ValueError('Image rank 4 is not supported')

  # Will keep track of the new altered bboxes after aug_func is repeatedly
  # applied. The -1 values are a dummy value and this first Tensor will be
  # removed upon appending the first real bbox.
  new_bboxes = tf.constant(_INVALID_BOX)

  # If the bboxes are empty, then just give it _INVALID_BOX. The result
  # will be thrown away.
  bboxes = tf.cond(tf.equal(tf.size(bboxes), 0),
                   lambda: tf.constant(_INVALID_BOX),
                   lambda: bboxes)

  bboxes = tf.ensure_shape(bboxes, (None, 4))

  # pylint:disable=g-long-lambda
  wrapped_aug_func = (
      lambda _image, bbox, _new_bboxes: _apply_bbox_augmentation_wrapper(
          _image, bbox, _new_bboxes, prob, aug_func, func_changes_bbox, *args))
  # pylint:enable=g-long-lambda

  # Setup the while_loop.
  num_bboxes = tf.shape(bboxes)[0]  # We loop until we go over all bboxes.
  idx = tf.constant(0)  # Counter for the while loop.

  # Conditional function when to end the loop once we go over all bboxes
  # images_and_bboxes contain (_image, _new_bboxes)
  cond = lambda _idx, _images_and_bboxes: tf.less(_idx, num_bboxes)

  # Shuffle the bboxes so that the augmentation order is not deterministic if
  # we are not changing the bboxes with aug_func.
  if not func_changes_bbox:
    loop_bboxes = tf.random.shuffle(bboxes)
  else:
    loop_bboxes = bboxes

  # Main function of while_loop where we repeatedly apply augmentation on the
  # bboxes in the image.
  # pylint:disable=g-long-lambda
  body = lambda _idx, _images_and_bboxes: [
      _idx + 1, wrapped_aug_func(_images_and_bboxes[0],
                                 loop_bboxes[_idx],
                                 _images_and_bboxes[1])]
  # pylint:enable=g-long-lambda

  _, (image, new_bboxes) = tf.while_loop(
      cond, body, [idx, (image, new_bboxes)],
      shape_invariants=[idx.get_shape(),
                        (image.get_shape(), tf.TensorShape([None, 4]))])

  # Either return the altered bboxes or the original ones depending on if
  # we altered them in anyway.
  if func_changes_bbox:
    final_bboxes = new_bboxes
  else:
    final_bboxes = bboxes
  return image, final_bboxes


def grayscale(image: tf.Tensor) -> tf.Tensor:
  if image.shape[-1] == 3:
    image = tf.image.rgb_to_grayscale(image)
  return image


def func_foreach_channel(func, image):
  """Applies a function separately to each channel of an image."""
  channels = tf.unstack(image, axis=-1)
  return tf.stack([func(channel) for channel in channels], axis=-1)


def wrap(image: tf.Tensor) -> tf.Tensor:
  """Returns 'image' with an extra channel set to all 1s."""
  shape = tf.shape(image)
  extended_channel = tf.expand_dims(tf.ones(shape[:-1], image.dtype), -1)
  return tf.concat([image, extended_channel], axis=-1)


def unwrap(image: tf.Tensor, replace: int) -> tf.Tensor:
  image_shape = tf.shape(image)
  # Flatten the spatial dimensions.
  flattened_image = tf.reshape(image, [-1, image_shape[-1]])
  # Find all pixels where the last channel is zero.
  alpha_channel = tf.expand_dims(flattened_image[..., -1], axis=-1)
  replace = tf.concat([replace, tf.ones([1], image.dtype)], 0)
  # Where they are zero, fill them in with 'replace'.
  flattened_image = tf.where(
      tf.equal(alpha_channel, 0),
      tf.ones_like(flattened_image, dtype=image.dtype) * replace,
      flattened_image)

  image = tf.reshape(flattened_image, image_shape)
  image = tf.slice(
      image,
      [0] * image.shape.rank,
      tf.concat([image_shape[:-1], [3]], -1))
  return image


def _normalize_tuple(value, n):
  if isinstance(value, int):
    return (value,) * n
  try:
    value_tuple = tuple(value)
  except TypeError as e:
    raise TypeError('`value` should be an integer or a tuple') from e
  if len(value_tuple) != n:
    raise ValueError('`value` should have length {}'.format(n))

  for single_value in value_tuple:
    try:
      int(single_value)
    except (TypeError, ValueError) as e:
      raise ValueError('All values in `value` should be integers') from e
  return value_tuple


def _get_gaussian_kernel(sigma, filter_shape):
  sigma = tf.convert_to_tensor(sigma)
  x = tf.range(-filter_shape // 2 + 1, filter_shape // 2 + 1)
  x = tf.cast(x ** 2, sigma.dtype)
  x = tf.nn.softmax(-x / (2 * sigma ** 2))
  return x


def _pad(image: tf.Tensor,
         filter_shape: Union[List[int], Tuple[int, ...]],
         mode: str = 'constant',
         constant_values: Union[int, tf.Tensor] = 0,
) -> tf.Tensor:
  if mode.upper() not in {'CONSTANT', 'REFLECT', 'SYMMETRIC'}:
    raise ValueError('Invalid mode: {}'.format(mode))
  constant_values = tf.convert_to_tensor(constant_values, dtype=image.dtype)
  filter_height, filter_width = filter_shape
  pad_top = (filter_height - 1) // 2
  pad_bottom = filter_height - 1 - pad_top
  pad_left = (filter_width - 1) // 2
  pad_right = filter_width - 1 - pad_left
  paddings = [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
  return tf.pad(image, paddings, mode=mode, constant_values=constant_values)


def gaussian_filter2d(image: tf.Tensor,
                      filter_shape: Union[List[int], Tuple[int, ...], int],
                      sigma: Union[List[float], Tuple[float], float] = 1.0,
                      padding: str = 'reflect',
                      constant_values: Union[int, tf.Tensor] = 0) -> tf.Tensor:
  if isinstance(sigma, (list, tuple)):
    if len(sigma) != 2:
      raise ValueError('If sigma is a list or tuple, it must contain exactly '
                       'two values.')
    else:
      sigma = (sigma,) * 2

  if any(s < 0 for s in sigma):
    raise ValueError('Sigma values should be non-negative.')

  image = tf.convert_to_tensor(image)
  sigma = tf.convert_to_tensor(sigma)
  image = tf.expand_dims(image, 0)
  # Keep the precision if it's float32, otherwise cast to float32.
  old_dtype = image.dtype
  if not image.dtype.is_floating:
    image = tf.cast(image, tf.float32)
  channels = tf.shape(image)[-1]
  filter_shape = _normalize_tuple(filter_shape, 2)
  sigma = tf.cast(sigma, image.dtype)
  gaussian_kernel_x = _get_gaussian_kernel(sigma[1], filter_shape[1])
  gaussian_kernel_x = gaussian_kernel_x[tf.newaxis, :]
  gaussian_kernel_y = _get_gaussian_kernel(sigma[0], filter_shape[0])
  gaussian_kernel_y = gaussian_kernel_y[:, tf.newaxis]
  gaussian_kernel_2d = tf.matmul(gaussian_kernel_x, gaussian_kernel_y)
  gaussian_kernel_2d = gaussian_kernel_2d[:, :, tf.newaxis, tf.newaxis]
  gaussian_kernel_2d = tf.tile(gaussian_kernel_2d, [1, 1, channels, 1])
  image = _pad(image, filter_shape, padding, constant_values)
  output = tf.nn.depthwise_conv2d(image,
                                  gaussian_kernel_2d,
                                  [1, 1, 1, 1],
                                  padding='VALID')
  return tf.cast(output, old_dtype)


def transform(image: tf.Tensor,
              transforms: Any,
              interpolation: str = 'nearest',
              output_shape=None,
              fill_mode: str = 'reflect',
              fill_value: float = 0.0) -> tf.Tensor:
  transforms = tf.convert_to_tensor(transforms, dtype=tf.float32)
  if transforms.shape.rank == 1:
    transforms = transforms[None]
  images = tf.expand_dims(image, 0)
  if output_shape is None:
    output_shape = tf.shape(images)[1:3]
    if not tf.executing_eagerly():
      output_shape_value = tf.get_static_value(output_shape)
      if output_shape_value is not None:
        output_shape = output_shape_value
  output_shape = tf.convert_to_tensor(output_shape, dtype=tf.int32)
  fill_value = tf.convert_to_tensor(fill_value, dtype=tf.float32)
  images = tf.raw_ops.ImageProjectiveTransformV3(
    images=images,
    output_shape=output_shape,
    fill_value=fill_value,
    fill_mode=fill_mode.upper(),
    transforms=transforms,
    interpolation=interpolation.upper(),
  )

  return tf.squeeze(images, 0)


def translate(image: tf.Tensor,
              translations,
              fill_value: float = 0.0,
              fill_mode: str = 'reflect',
              interpolation: str = 'nearest') -> tf.Tensor:
  translations = tf.convert_to_tensor(translations, dtype=tf.float32)
  if translations.get_shape().ndims is None:
    raise TypeError('translations rank must be statically known')
  elif len(translations.get_shape()) == 1:
    translations = translations[None]
  elif len(translations.get_shape()) != 2:
    raise TypeError('translations should have rank 1 or 2.')
  num_translations = tf.shape(translations)[0]
  transforms = tf.concat(
      [
        tf.ones((num_translations, 1), tf.float32),
        tf.zeros((num_translations, 1), tf.float32),
        -translations[:, 0, None],
        tf.zeros((num_translations, 1), tf.float32),
        tf.ones((num_translations, 1), tf.float32),
        -translations[:, 1, None],
        tf.zeros((num_translations, 2), tf.float32),
      ],
      axis=1,
  )
  return transform(
      image,
      transforms=transforms,
      fill_value=fill_value,
      fill_mode=fill_mode,
      interpolation=interpolation,
  )


def blend(image1: tf.Tensor, image2: tf.Tensor, factor: float) -> tf.Tensor:
  """Blend image1 and image2 using 'factor'."""
  if factor == 0.0:
    return tf.convert_to_tensor(image1)
  if factor == 1.0:
    return tf.convert_to_tensor(image2)

  image1 = tf.cast(image1, tf.float32)
  image2 = tf.cast(image2, tf.float32)
  difference = image2 - image1
  scaled = factor * difference
  temp = tf.cast(image1, tf.float32) + scaled
  if factor > 0.0 and factor < 1.0:
    return tf.cast(temp, tf.uint8)
  return tf.cast(tf.clip_by_value(temp, 0, 255), tf.uint8)


def autocontrast(image: tf.Tensor) -> tf.Tensor:
  """Implements Autocontrast function from PIL using TF ops."""
  def scale_channel(image: tf.Tensor) -> tf.Tensor:
    # Scale the 2D image using the autocontrast rule.
    lo = tf.cast(tf.reduce_min(image), tf.float32)
    hi = tf.cast(tf.reduce_max(image), tf.float32)
    # Scale the image, making the lowest value 0 and the highest value 255.
    def scale_values(_image):
      scale = 255.0 / (hi - lo)
      offset = -lo * scale
      _image = tf.cast(_image, tf.float32) * scale + offset
      _image = tf.clip_by_value(_image, 0.0, 255.0)
      return tf.cast(_image, tf.uint8)
    result = tf.cond(hi > lo, lambda: scale_values(image), lambda: image)
    return result

  return func_foreach_channel(scale_channel, image)


def equalize(image: tf.Tensor) -> tf.Tensor:
  """Implements Equalize function from PIL using TF ops."""
  def scale_channel(image):
    # Scale the data in the channel to implement equalize.
    image = tf.cast(image, tf.int32)
    histo = tf.histogram_fixed_width(image, [0, 255], nbins=256)
    # Filter out the nonzeros
    nonzero = tf.where(tf.not_equal(histo, 0))
    nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
    step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255

    def build_lut(histo, step):
      # Compute the cumulative sum, shifting by step // 2
      # and then normalization by step.
      lut = (tf.cumsum(histo) + (step // 2)) // step
      # Shift lut, prepending with 0.
      lut = tf.concat([[0], lut[:-1]], 0)
      # Clip the counts to be in range. This is done
      # in the C code for image.point.
      return tf.clip_by_value(lut, 0, 255)

    result = tf.cond(tf.equal(step, 0),
                     lambda: image,
                     lambda: tf.gather(build_lut(histo, step), image))
    return tf.cast(result, tf.uint8)
  return func_foreach_channel(scale_channel, image)


def invert(image: tf.Tensor) -> tf.Tensor:
  """Inverts the input image."""
  return 255 - image


def rotate(image: tf.Tensor, degrees: float) -> tf.Tensor:
  degrees_to_radians = math.pi / 180.0
  radians = tf.cast(degrees * degrees_to_radians, tf.float32)

  image_height = tf.cast(tf.shape(image)[0], tf.float32)
  image_width = tf.cast(tf.shape(image)[1], tf.float32)

  angles = tf.convert_to_tensor(radians, dtype=tf.float32)
  if len(angles.get_shape()) == 0:
    angles = angles[None]
  elif len(angles.get_shape()) != 1:
    raise TypeError('Angles should have a rank 0 or 1.')

  x_offset = ((image_width - 1) -
              (tf.math.cos(angles) * (image_width - 1) - tf.math.sin(angles) *
               (image_height - 1))) / 2.0
  y_offset = ((image_height - 1) -
              (tf.math.sin(angles) * (image_width - 1) + tf.math.cos(angles) *
               (image_height - 1))) / 2.0
  num_angles = tf.shape(angles)[0]
  transforms = tf.concat(
      [
          tf.math.cos(angles)[:, None],
          -tf.math.sin(angles)[:, None],
          x_offset[:, None],
          tf.math.sin(angles)[:, None],
          tf.math.cos(angles)[:, None],
          y_offset[:, None],
          tf.zeros([num_angles, 2], tf.float32),
      ],
      axis=1,
  )
  image = transform(image, transforms=transforms)
  return image


def wrapped_rotate(image: tf.Tensor,
                   degrees: float,
                   replace: int) -> tf.Tensor:
  image = rotate(wrap(image), degrees=degrees)
  return unwrap(image, replace)


def posterize(image: tf.Tensor, bits: int) -> tf.Tensor:
  """Equivalent of PIL Posterize."""
  shift = 8 - bits
  return tf.bitwise.left_shift(tf.bitwise.right_shift(image, shift), shift)


def solarize(image: tf.Tensor,
             threshold: int = 128) -> tf.Tensor:
  """Solarize the input image(s)."""
  # For each pixel in the image, select the pixel
  # if the value is less than the threshold.
  # Otherwise, subtract 255 from the pixel.
  return tf.where(image < threshold, image, 255 - image)


def solarize_add(image: tf.Tensor,
                 addition: int = 0,
                 threshold: int = 128) -> tf.Tensor:
  """Additive solarize the input image(s)."""
  # For each pixel in the image less than threshold
  # we add 'addition' amount to it and then clip the
  # pixel value to be between 0 and 255. The value
  # of 'addition' is between -128 and 128.
  added_image = tf.cast(image, tf.int64) + addition
  added_image = tf.cast(tf.clip_by_value(added_image, 0, 255), tf.uint8)
  return tf.where(image < threshold, added_image, image)


def color(image: tf.Tensor, factor: float) -> tf.Tensor:
  """Equivalent of PIL Color."""
  num_channel = image.shape[-1]
  degenerate = grayscale(image)
  if num_channel == 3:
    degenerate = tf.image.grayscale_to_rgb(degenerate)
  return blend(degenerate, image, factor)


def contrast(image: tf.Tensor, factor: float) -> tf.Tensor:
  """Equivalent of PIL Contrast."""
  num_channel = image.shape[-1]
  degenerate = grayscale(image)
  # Cast before calling tf.histogram.
  degenerate = tf.cast(degenerate, tf.int32)

  # Compute the grayscale histogram, then compute the mean pixel value,
  # and create a constant image size of that value.  Use that as the
  # blending degenerate target of the original image.
  hist = tf.histogram_fixed_width(degenerate, [0, 255], nbins=256)
  mean = tf.reduce_sum(tf.cast(hist, tf.float32)) / 256.0
  degenerate = tf.ones_like(degenerate, dtype=tf.float32) * mean
  degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
  if num_channel == 3:
    degenerate = tf.image.grayscale_to_rgb(tf.cast(degenerate, tf.uint8))
  return blend(degenerate, image, factor)


def brightness(image: tf.Tensor, factor: float) -> tf.Tensor:
  """Equivalent of PIL Brightness."""
  degenerate = tf.zeros_like(image)
  return blend(degenerate, image, factor)


def sharpness(image: tf.Tensor, factor: float) -> tf.Tensor:
  orig_image = image
  image = tf.cast(image, tf.float32)
  image = tf.expand_dims(image, 0)
  if orig_image.shape.rank == 3:
    kernel = tf.constant([[1, 1, 1], [1, 5, 1], [1, 1, 1]],
                         dtype=tf.float32,
                         shape=[3, 3, 1, 1]) / 13.
    # Tile across channel dimension.
    kernel = tf.tile(kernel, [1, 1, 3, 1])
    strides = [1, 1, 1, 1]
    degenerate = tf.nn.depthwise_conv2d(
        image, kernel, strides, padding='VALID', dilations=[1, 1])
  elif orig_image.shape.rank == 4:
    kernel = tf.constant([[1, 1, 1], [1, 5, 1], [1, 1, 1]],
                         dtype=tf.float32,
                         shape=[1, 3, 3, 1, 1]) / 13.
    strides = [1, 1, 1, 1, 1]
    # Run the kernel across each channel
    degenerate = func_foreach_channel(
        lambda x: tf.nn.conv3d(x, kernel, strides, padding='VALID',
                               dilations=[1, 1, 1, 1, 1]), image)
  else:
    raise ValueError('Bad image rank: {}'.format(image.shape.rank))

  degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
  degenerate = tf.squeeze(tf.cast(degenerate, tf.uint8), [0])

  # For the borders of the resulting image, fill in the values of the
  # original image.
  mask = tf.ones_like(degenerate)
  paddings = [[0, 0]] * (orig_image.shape.rank - 3)
  padded_mask = tf.pad(mask, paddings + [[1, 1], [1, 1], [0, 0]])
  padded_degenerate = tf.pad(degenerate, paddings + [[1, 1], [1, 1], [0, 0]])
  result = tf.where(tf.equal(padded_mask, 1), padded_degenerate, orig_image)

  # Blend the final result.
  return blend(result, orig_image, factor)


def shear_x(image: tf.Tensor, level: float, replace: int) -> tf.Tensor:
  """Equivalent of PIL Shearing in X dimension."""
  # Shear parallel to x axis is a projective transform
  # with a matrix form of:
  # [1  level
  #  0  1].
  image = transform(
      image=wrap(image), transforms=[1., level, 0., 0., 1., 0., 0., 0.])
  return unwrap(image, replace)


def shear_y(image: tf.Tensor, level: float, replace: int) -> tf.Tensor:
  """Equivalent of PIL Shearing in Y dimension."""
  # Shear parallel to y axis is a projective transform
  # with a matrix form of:
  # [1  0
  #  level  1].
  image = transform(
      image=wrap(image), transforms=[1., 0., 0., level, 1., 0., 0., 0.])
  return unwrap(image, replace)


def translate_x(image: tf.Tensor, pixels: int, replace: int) -> tf.Tensor:
  """Equivalent of PIL Translate in X dimension."""
  image = translate(wrap(image), [-pixels, 0])
  return unwrap(image, replace)


def translate_y(image: tf.Tensor, pixels: int, replace: int) -> tf.Tensor:
  """Equivalent of PIL Translate in Y dimension."""
  image = translate(wrap(image), [0, -pixels])
  return unwrap(image, replace)


def cutout(image: tf.Tensor, pad_size: int, replace: int = 0) -> tf.Tensor:
  """Apply cutout (https://arxiv.org/abs/1708.04552) to image.

  This operation applies a (2*pad_size x 2*pad_size) mask of zeros to
  a random location within `image`. The pixel values filled in will be of the
  value `replace`. The location where the mask will be applied is randomly
  chosen uniformly over the whole image.
  """
  if image.shape.rank not in [3, ]:
    raise ValueError('Bad image rank: {}'.format(image.shape.rank))

  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]

  # Sample the center location in the image where the zero mask will be applied.
  cutout_center_height = tf.random.uniform(
      shape=[], minval=0, maxval=image_height, dtype=tf.int32)

  cutout_center_width = tf.random.uniform(
      shape=[], minval=0, maxval=image_width, dtype=tf.int32)

  image = _fill_rectangle(image, cutout_center_width, cutout_center_height,
                          pad_size, pad_size, replace)
  return image


def rotate_with_bboxes(image, bboxes, degrees, replace):
  """Equivalent of PIL Rotate that rotates the image and bbox."""
  if image.shape.rank == 4:
    raise ValueError('Image rank 4 is not supported')

  # Rotate the image.
  image = wrapped_rotate(image, degrees, replace)

  # Convert bbox coordinates to pixel values.
  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]
  # pylint:disable=g-long-lambda
  wrapped_rotate_bbox = lambda bbox: _rotate_bbox(
      bbox, image_height, image_width, degrees)
  # pylint:enable=g-long-lambda
  bboxes = tf.map_fn(wrapped_rotate_bbox, bboxes)
  return image, bboxes


def gaussian_noise(image: tf.Tensor,
                   low: float = 0.1,
                   high: float = 2.0) -> tf.Tensor:
  augmented_image = gaussian_filter2d(  # pylint: disable=g-long-lambda
      image, filter_shape=[3, 3], sigma=np.random.uniform(low=low, high=high)
  )
  return augmented_image


def shear_with_bboxes(image: tf.Tensor,
                      bboxes: Optional[tf.Tensor],
                      level: float,
                      replace: int,
                      shear_horizontal: bool) -> tf.Tensor:
  if image.shape.rank == 4:
    raise ValueError('Image rank 4 is not supported')

  if shear_horizontal:
    image = shear_x(image, level, replace)
  else:
    image = shear_y(image, level, replace)

  # Convert bbox coordinates to pixel values.
  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]
  # pylint:disable=g-long-lambda
  wrapped_shear_bbox = lambda bbox: _shear_bbox(
      bbox, image_height, image_width, level, shear_horizontal)
  # pylint:enable=g-long-lambda
  bboxes = tf.map_fn(wrapped_shear_bbox, bboxes)
  return image, bboxes


def translate_bbox(image: tf.Tensor,
                   bboxes: Optional[tf.Tensor],
                   pixels: int,
                   replace: int,
                   shift_horizontal: bool) -> tf.Tensor:
  if image.shape.rank == 4:
    raise ValueError('Image rank 4 is not supported')

  if shift_horizontal:
    image = translate_x(image, pixels, replace)
  else:
    image = translate_y(image, pixels, replace)

  # Convert bbox coordinates to pixel values.
  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]
  # pylint:disable=g-long-lambda
  wrapped_shift_bbox = lambda bbox: _shift_bbox(
      bbox, image_height, image_width, pixels, shift_horizontal)
  # pylint:enable=g-long-lambda
  bboxes = tf.map_fn(wrapped_shift_bbox, bboxes)
  return image, bboxes


def translate_y_only_bboxes(image: tf.Tensor,
                            bboxes: Optional[tf.Tensor],
                            prob: float,
                            pixels: int,
                            replace: int) -> tf.Tensor:
  if bboxes.shape.rank == 4:
    raise ValueError('Bbox rank 4 is not supported')

  func_changes_bbox = False
  prob = prob / 3.0
  num_bboxes = tf.shape(bboxes)[0]
  # pylint:disable=g-long-lambda
  image, bboxes = tf.cond(
      tf.equal(num_bboxes, 0),
      lambda: (image, bboxes),
      lambda: _apply_multi_bbox_augmentation(
          image, bboxes, prob, pixels, replace, func_changes_bbox))
  # pylint:enable=g-long-lambda
  return image, bboxes


def name_to_func(name):
  """Converts the string name to a function."""
  return {
    'AutoContrast': autocontrast,
    'Equalize': equalize,
    'Invert': invert,
    'Rotate': wrapped_rotate,
    'Posterize': posterize,
    'Solarize': solarize,
    'SolarizeAdd': solarize_add,
    'Color': color,
    'Contrast': contrast,
    'Brightness': brightness,
    'Sharpness': sharpness,
    'ShearX': shear_x,
    'ShearY': shear_y,
    'TranslateX': translate_x,
    'TranslateY': translate_y,
    'Cutout': cutout,
    'Rotate_BBox': rotate_with_bboxes,
    'Grayscale': grayscale,
    'Gaussian_Noise': gaussian_noise,
    # pylint:disable=g-long-lambda
    'ShearX_BBox': lambda image, bboxes, level, replace: shear_with_bboxes(
        image, bboxes, level, replace, shear_horizontal=True),
    'ShearY_BBox': lambda image, bboxes, level, replace: shear_with_bboxes(
        image, bboxes, level, replace, shear_horizontal=False),
    'TranslateX_BBox': lambda image, bboxes, pixels, replace: translate_bbox(
        image, bboxes, pixels, replace, shift_horizontal=True),
    'TranslateY_BBox': lambda image, bboxes, pixels, replace: translate_bbox(
        image, bboxes, pixels, replace, shift_horizontal=False),
    # pylint:enable=g-long-lambda
    'TranslateY_Only_BBoxes': translate_y_only_bboxes,
  }[name]


def _randomly_negate_tensor(tensor):
  should_flip = tf.cast(tf.floor(tf.random.uniform([]) + 0.5), tf.bool)
  final_tensor = tf.cond(should_flip, lambda: tensor, lambda: -tensor)
  return final_tensor


def _translate_level_to_arg(level: float, translate_const: float):
  level = (level / _MAX_LEVEL) * float(translate_const)
  level = _randomly_negate_tensor(level)
  return (level,)


def _rotate_level_to_arg(level: float):
  level = (level / _MAX_LEVEL) * 30.
  level = _randomly_negate_tensor(level)
  return (level,)


def _shear_level_to_arg(level: float):
  level = (level / _MAX_LEVEL) * 0.3
  level = _randomly_negate_tensor(level)
  return (level,)


def _gaussian_noise_level_to_arg(level: float, translate_const: float):
  low_std = (level / _MAX_LEVEL)
  high_std = translate_const * low_std
  return low_std, high_std


def level_to_arg(cutout_const: float, translate_const: float):
  """Creates a dict mapping image operation names to their arguments."""
  mult_to_arg = lambda level, multiplier: (
      int((level / _MAX_LEVEL) * multiplier), )
  enhance_level_to_arg = lambda level: ((level / _MAX_LEVEL) * 1.8 + 0.1, )

  no_arg = lambda _: ()
  posterize_arg = lambda level: mult_to_arg(level, 4)
  solarize_arg = lambda level: mult_to_arg(level, 256)
  solarize_add_arg = lambda level: mult_to_arg(level, 110)
  cutout_arg = lambda level: mult_to_arg(level, cutout_const)
  translate_arg = lambda level: _translate_level_to_arg(level, translate_const)
  translate_bbox_arg = lambda level: _translate_level_to_arg(level, 120)

  return {
      'AutoContrast': no_arg,
      'Equalize': no_arg,
      'Invert': no_arg,
      'Rotate': _rotate_level_to_arg,
      'Posterize': posterize_arg,
      'Solarize': solarize_arg,
      'SolarizeAdd': solarize_add_arg,
      'Color': enhance_level_to_arg,
      'Contrast': enhance_level_to_arg,
      'Brightness': enhance_level_to_arg,
      'Sharpness': enhance_level_to_arg,
      'ShearX': _shear_level_to_arg,
      'ShearY': _shear_level_to_arg,
      'Cutout': cutout_arg,
      'TranslateX': translate_arg,
      'TranslateY': translate_arg,
      'Rotate_BBox': _rotate_level_to_arg,
      'ShearX_BBox': _shear_level_to_arg,
      'ShearY_BBox': _shear_level_to_arg,
      'Grayscale': no_arg,
      # pylint:disable=g-long-lambda
      'Gaussian_Noise': lambda level: _gaussian_noise_level_to_arg(
          level, translate_const),
      # pylint:disable=g-long-lambda
      'TranslateX_BBox': lambda level: _translate_level_to_arg(
          level, translate_const),
      'TranslateY_BBox': lambda level: _translate_level_to_arg(
          level, translate_const),
      # pylint:enable=g-long-lambda
      'TranslateY_Only_BBoxes': translate_bbox_arg,
  }


def bbox_wrapper(func):
  """Adds a bboxes function argument to func and returns unchanged bboxes."""
  def wrapper(images, bboxes, *args, **kwargs):
    return (func(images, *args, **kwargs), bboxes)
  return wrapper


def parse_policy_info(name: str,
                      prob: float,
                      level: float,
                      replace_value: List[int],
                      cutout_const: float,
                      translate_const: float,
                      level_std: float = 0.) -> Tuple[Any, float, Any]:
  """Return the function that corresponds to `name` and update `level` param."""
  func = name_to_func(name)

  if level_std > 0:
    level += tf.random.normal([], dtype=tf.float32)
    level = tf.clip_by_value(level, 0., _MAX_LEVEL)

  args = level_to_arg(cutout_const, translate_const)[name](level)

  if name in PROB_FUNCS:
    # Add in the prob arg if it is required for the function that is called.
    args = tuple([prob] + list(args))

  if name in REPLACE_FUNCS:
    # Add in replace arg if it is required for the function that is called.
    args = tuple(list(args) + [replace_value])

  # Add bboxes as the second positional argument for the function if it does
  # not already exist.
  if 'bboxes' not in inspect.getfullargspec(func)[0]:
    func = bbox_wrapper(func)

  return func, prob, args

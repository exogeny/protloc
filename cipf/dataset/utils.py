import io
import os
import cv2
import uuid
import dataclasses

import numpy as np

from PIL import Image
import tensorflow_datasets as tfds


@dataclasses.dataclass
class SubcellularLocation:
  gene_id: str
  gene_name: str
  reliability: str
  main_location: str
  additional_location: str
  extracellular_location: str


subcellular_location_tree = {
    'Nuclear membrane':           [0,  0,  0],
    'Nucleoli':                   [0,  1,  1],
    'Nucleoli fibrillar center':  [0,  1,  2],
    'Kinetochore':                [0,  2,  3],
    'Mitotic chromosome':         [0,  2,  4],
    'Nuclear bodies':             [0,  2,  5],
    'Nuclear speckles':           [0,  2,  6],
    'Nucleoplasm':                [0,  2,  7],
    'Actin filaments':            [1,  3,  8],
    'Cleavage furrow':            [1,  3,  9],
    'Focal adhesion sites':       [1,  3, 10],
    'Centriolar satellite':       [1,  4, 11],
    'Centrosome':                 [1,  4, 12],
    'Aggresome':                  [1,  5, 13],
    'Cytoplasmic bodies':         [1,  5, 14],
    'Cytosol':                    [1,  5, 15],
    'Rods & Rings':               [1,  5, 33],
    'Intermediate filaments':     [1,  6, 16],
    'Cytokinetic bridge':         [1,  7, 17],
    'Microtubule ends':           [1,  7, 18],
    'Microtubules':               [1,  7, 19],
    'Midbody':                    [1,  7, 20],
    'Midbody ring':               [1,  7, 21],
    'Mitotic spindle':            [1,  7, 22],
    'Mitochondria':               [1,  8, 23],
    'Endoplasmic reticulum':      [2,  9, 24],
    'Golgi apparatus':            [2, 10, 25],
    'Cell Junctions':             [2, 11, 26],
    'Plasma membrane':            [2, 11, 27],
    'Endosomes':                  [2, 12, 28],
    'Lipid droplets':             [2, 12, 29],
    'Lysosomes':                  [2, 12, 30],
    'Peroxisomes':                [2, 12, 31],
    'Vesicles':                   [2, 12, 32],
}


def read_hpa_image(brg_image_path, y_image_path):
  brg_image_path = str(brg_image_path)
  y_image_path = str(y_image_path)
  image1 = cv2.imread(brg_image_path)
  image2 = cv2.imread(y_image_path, cv2.IMREAD_GRAYSCALE)
  image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
  image2 = np.expand_dims(image2, axis=2)
  image = np.concatenate((image1, image2), axis=2)
  return image


def serialize_image(image, mode):
  with io.BytesIO() as output:
    Image.fromarray(image, mode=mode).save(output, format='PNG')
    return output.getvalue()


def generate_examples_from_image(image, segmentation_path, split_name=False):
  segmentation_path = str(segmentation_path)
  segmentation = cv2.imread(segmentation_path, cv2.IMREAD_GRAYSCALE)
  segmentation[segmentation > 40] = 255
  _, labels, stats, _ = cv2.connectedComponentsWithStats(segmentation)

  bboxes, indices = [], []
  for i, stat in enumerate(stats[1:]):
    x, y, w, h, area = stat
    if area < 32 * 32 or w < 32 or h < 32:
      continue

    indices.append(i + 1)
    bboxes.append([x, y, w, h])

  train_count = int(len(bboxes) * 0.7)
  if split_name == tfds.Split.TRAIN:
    bboxes = bboxes[:train_count]
    indices = indices[:train_count]
  else:
    bboxes = bboxes[train_count:]
    indices = indices[train_count:]

  for i, (x, y, w, h) in zip(indices, bboxes):
    patch = image[y:y+h, x:x+w]
    segmentation = labels == i
    segmentation = np.array(segmentation, dtype=np.uint8)
    segmentation = segmentation[y:y+h, x:x+w]

    yield {
        'image': serialize_image(patch, 'RGBA'),
        'image/mask': serialize_image(segmentation, 'L'),
    }


def get_subcellular_locations(subcellular_location_file):
  subcellular_location_lines = open(subcellular_location_file, 'r').readlines()[1:]
  subcellular_location_lines = [line.strip().split('\t') for line in subcellular_location_lines]
  gene_names = set()
  subcellular_location_lines_filtered = []
  for line in subcellular_location_lines:
    if line[1] not in gene_names:
      gene_names.add(line[1])
      subcellular_location_lines_filtered.append(line)

  subcellular_locations = {
    line[0]: SubcellularLocation(
      gene_id=line[0],
      gene_name=line[1],
      reliability=line[2],
      main_location=line[3],
      additional_location=line[4],
      extracellular_location=line[5],
    ) for line in subcellular_location_lines_filtered
  }
  return subcellular_locations


def get_location_ids(location):
  def get_location_id(location_str):
    level1 = [0] * 3
    level2 = [0] * 13
    level3 = [0] * 34

    for l in location_str.split(';'):
      if len(l) == 0 or l not in subcellular_location_tree:
        continue
      indices = subcellular_location_tree[l]
      level1[indices[0]] = 1
      level2[indices[1]] = 1
      level3[indices[2]] = 1
    return level1, level2, level3

  main1, main2, main3 = get_location_id(location.main_location)
  additional1, additional2, additional3 = get_location_id(location.additional_location)
  extracellular1, extracellular2, extracellular3 = get_location_id(location.extracellular_location)

  return {
    'location/main/level1': main1,
    'location/main/level2': main2,
    'location/main/level3': main3,
    'location/additional/level1': additional1,
    'location/additional/level2': additional2,
    'location/additional/level3': additional3,
    'location/extracellular/level1': extracellular1,
    'location/extracellular/level2': extracellular2,
    'location/extracellular/level3': extracellular3,
    'location/reliability': location.reliability,
  }


def generate_examples_for_human_protein_atlas(assay_id,
                                              image_id,
                                              gene_id,
                                              gene_name,
                                              split_name,
                                              image_paths,
                                              subcellular_locations):
  key = f'{assay_id}-{image_id}'
  image = read_hpa_image(
      image_paths[f'{key}-c'], image_paths[f'{key}-y'])
  location = subcellular_locations.get(gene_id, None)
  location_ids = get_location_ids(location)
  examples = []
  for example in generate_examples_from_image(image,
                                              image_paths[f'{key}-s'],
                                              split_name):
    example.update({
      'ensg/id': gene_id,
      'ensg/name': gene_name,
    })
    index = str(uuid.uuid4())
    example.update(location_ids)
    examples.append((f'{key}-{index}', example))
  return examples
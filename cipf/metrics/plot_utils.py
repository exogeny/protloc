import io
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt


def get_figure(fig) -> np.ndarray:
  fig.tight_layout()
  buffer = io.BytesIO()
  fig.savefig(buffer, format='png', dpi=300)
  buffer.seek(0)
  image = np.array(Image.open(buffer))
  buffer.close()
  return image


def plot_confusion_matrix(cm):
  dfcm = pd.DataFrame(cm)
  plt.figure(figsize=(7, 7))
  fig = sns.heatmap(dfcm, annot=True, vmin=0.0, vmax=1.0).get_figure()
  image = get_figure(fig)
  plt.close()
  return image


def plot_multilabel_confusion_matrix(cm):
  cm = cm.reshape((-1, 4))
  dfcm = pd.DataFrame(cm, columns=['tp', 'fp', 'fn', 'tn'])
  plt.figure(figsize=(4, 1 * cm.shape[0]))
  fig = sns.heatmap(dfcm, annot=True, vmin=0.0, vmax=1.0).get_figure()
  image = get_figure(fig)
  plt.close()
  return image

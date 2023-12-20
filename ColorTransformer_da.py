import numpy as np

class ColorTransformer:
  def __init__(self, mode='potsdam'):
    # color table.
    self.mode = mode
    self.clr_tab = self.createColorTable()
    # id table.
    id_tab = {}
    for k, v in self.clr_tab.items():
      id_tab[k] = self.clr2id(v)
    self.id_tab = id_tab

  def createColorTable(self):
    clr_tab = {}
    if self.mode == 'potsdam' or self.mode == 'vaihingen':
      clr_tab['ImSurf'] = [255, 255, 255]
      clr_tab['Building'] = [0, 0, 255]
      clr_tab['LowVeg'] = [0, 255, 255]
      clr_tab['Tree'] = [0, 255, 0]
      clr_tab['Car'] = [255, 255, 0] # [255, 204, 0]
      clr_tab['Clutter'] = [255, 0, 0]

    if self.mode == 'uavid':
      clr_tab['Clutter'] = [0, 0, 0]
      clr_tab['Building'] = [128, 0, 0]
      clr_tab['Road'] = [128, 64, 128]
      clr_tab['Static_Car'] = [192, 0, 192]
      clr_tab['Tree'] = [0, 128, 0]
      clr_tab['Vegetation'] = [128, 128, 0]
      clr_tab['Human'] = [64, 64, 0]
      clr_tab['Moving_Car'] = [64, 0, 128]

    if self.mode == 'potsdam_t':
      clr_tab['ImSurf'] = [255, 255, 255] # 0-0
      clr_tab['Building'] = [0, 0, 255] # 1-1
      clr_tab['LowVeg'] = [0, 255, 255] # 2-4
      clr_tab['Tree'] = [0, 255, 0] # 3-2
      clr_tab['Car'] = [255, 204, 0] # 4-3
      clr_tab['Clutter'] = [255, 0, 0] # 5-4


    return clr_tab

  def colorTable(self):
    return self.clr_tab

  def clr2id(self, clr):
    return clr[0] + clr[1] * 255 + clr[2] * 255 * 255

  # transform to uint8 integer label
  def transform(self, label, dtype=np.int32):
    height, width = label.shape[:2]
    # default value is index of clutter.
    newLabel = np.ones((height, width), dtype=dtype) * dtype(5)
    id_label = label.astype(np.int64)
    id_label = id_label[:, :, 0] + id_label[:, :, 1] * 255 + id_label[:, :, 2] * 255 * 255
    for tid, val in enumerate(self.id_tab.values()):
      mask = (id_label == val)
      newLabel[mask] = tid
    return newLabel

  # transform back to 3 channels uint8 label
  def inverse_transform(self, label):
    label_img = np.zeros(shape=(label.shape[0], label.shape[1], 3), dtype=np.uint8)
    values = list(self.clr_tab.values())
    for tid, val in enumerate(values):
      mask = (label == tid)
      label_img[mask] = val
    return label_img

  def lab2rgb(self, lab):
    values = list(self.clr_tab.values())
    for tid, val in enumerate(values):
      if lab == tid:
        return val



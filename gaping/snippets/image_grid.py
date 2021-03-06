from . import texture_packer

from PIL import Image, ImageDraw

from pprint import pprint as pp

from .. import tf_tools as tft

def label_images(images, labels):
  if labels is None:
    return images
  r = []
  for label, image in zip(labels, images):
    draw = ImageDraw.Draw(image)
    if isinstance(label, bytes):
      label = label.decode('utf8')
    draw.text((2, 2), str(label), (255, 128, 255))
    r.append(image)
  return r


def save_grid(images, fname='test.jpg', quality=90, labels=None):
  tp = texture_packer.TexturePacker(one_pixel_border=True, force_power_of_two=True, width=4096, height=4096);
  images = [tft.bytes_to_pil(x) for x in images];
  images = label_images(images, labels)
  tp.textures.extend([texture_packer.Texture(*x.size) for x in images]);
  area, w, h = tp.pack_textures();
  w, h; image = Image.new('RGBA', (w, h));
  draw = ImageDraw.Draw(image);
  pp([x if x.placed else None for x in tp.get_texture_locations()])
  [image.alpha_composite((img.transpose(Image.TRANSPOSE).transpose(Image.FLIP_TOP_BOTTOM) if tex.flipped else img).convert('RGBA'), (tex.x, tex.y)) for tex, img in zip(tp.get_texture_locations(), images) if tex.placed]
  image.convert('RGB').save(fname, quality=quality)
  return image

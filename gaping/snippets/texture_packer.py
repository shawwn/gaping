from dataclasses import dataclass, field
from typing import List

@dataclass
class Rect:
  x1: int = 0
  y1: int = 0
  x2: int = 0
  y2: int = 0
  
  def intersects(self, r):
    if self.x2 < r.x1 or \
       self.x1 > r.x2 or \
       self.y2 < r.y1 or \
       self.y1 > r.y2:
      return False
    return True

  def to_list(self):
    return [(self.x1, self.y1), (self.x2, self.y2)]


@dataclass
class Texture:
  width: int
  height: int
  x: int = 0
  y: int = 0
  flipped: bool = False # true if the texture was rotated to make it fit better.
  placed: bool = False

  def place(self, x, y, flipped):
    self.x = x
    self.y = y
    self.flipped = flipped
    self.placed = True

  @property
  def area(self):
    return self.width * self.height

  @property
  def longest_edge(self):
    if self.width >= self.height:
      return self.width
    else:
      return self.height

  @property
  def rect(self):
    return Rect(self.x, self.y, self.x+self.width-1, self.y+self.height-1)

@dataclass
class Node:
  x: int
  y: int
  width: int
  height: int
  next: object = None
  
  def fits(self, width, height):
    ret = False
    edge_count = 0
    if width == self.width or \
       height == self.height or \
       width == self.height or \
       height == self.width:
      if width == self.width:
        edge_count += 1
        if height == self.height:
          edge_count += 1
      elif width == self.height:
        edge_count += 1
        if height == self.width:
          edge_count += 1
      elif height == self.width:
        edge_count += 1
      elif height == self.height:
        edge_count += 1

    if width <= self.width and height <= self.height:
      ret = True;
    elif height <= self.width and width <= self.height:
      ret = True;
    
    return ret, edge_count

  @property
  def rect(self):
    return Rect(self.x, self.y, self.x+self.width-1, self.y+self.height-1)

  def validate(self, node):
    if self.rect.intersects(node.rect):
      import pdb; pdb.set_trace()
    assert not self.rect.intersects(node.rect)

  def merge(self, node):
    ret = False;

    r1 = self.rect
    r2 = node.rect

    r1.x2 += 1
    r1.y2 += 1
    r2.x2 += 1
    r2.y2 += 1

    #return r1, r2

    # if we share the top edge then..
    if r1.x1 == r2.x1 and r1.x2 == r2.x2 and r1.y1 == r2.y2:
      y = node.y;
      self.height+=node.height;
      ret = True;
    elif r1.x1 == r2.x1 and r1.x2 == r2.x2 and r1.y2 == r2.y1: # if we share the bottom edge
      self.height+=node.height;
      ret = True;
    elif r1.y1 == r2.y1 and r1.y2 == r2.y1 and r1.x1 == r2.x2: # if we share the left edge
      x = node.x;
      self.width+=node.width;
      ret = True;
    elif r1.y1 == r2.y1 and r1.y2 == r2.y1 and r1.x2 == r2.x1: # if we share the left edge
      self.width+=node.width;
      ret = True;

    return ret;
    

def nextPow2(v):
  p = 1
  while p < v:
    p = p * 2
  return p

@dataclass
class TexturePacker:
  force_power_of_two: bool = False
  one_pixel_border: bool = False
  width: int = 0
  height: int = 0
  textures: List[Texture] = field(default_factory=list)
  free: Node = None

  def new_free(self, x, y, width, height):
    node = Node(x, y, width, height)
    node.next = self.free
    self.free = node

  def pack_textures(self):
    width = self.width
    height = self.height
    longest_edge = self.calc_longest_edge()

    if self.one_pixel_border:
      for t in self.textures:
        t.width+=2;
        t.height+=2;
      longest_edge+=2;

    if self.force_power_of_two:
      longest_edge = nextPow2(longest_edge);
    

    if width <= 0:
      width  = longest_edge;              # The width is no more than the longest edge of any rectangle passed in
    count = self.calc_total_area(placed=False) // (longest_edge*longest_edge);
    if height <= 0:
      height = (count+3)*longest_edge;            # We guess that the height is no more than twice the longest edge.  On exit, this will get shrunk down to the actual tallest height.

    self.new_free(0,0,width,height)

    # We must place each texture
    for i in range(len(self.textures)):
      index = 0
      longest_edge = 0
      most_area = 0

      # We first search for the texture with the longest edge, placing it first.
      # And the most area...
      for j in range(len(self.textures)):
        t = self.textures[j]
        if not t.placed: # if this texture has not already been placed.
          if t.longest_edge > longest_edge:
            most_area = t.area
            longest_edge = t.longest_edge
            index = j
          elif t.longest_edge == longest_edge: # if the same length, keep the one with the most area.
            if t.area > most_area:
              most_area = t.area
              index = j
      # For the texture with the longest edge we place it according to this criteria.
      #   (1) If it is a perfect match, we always accept it as it causes the least amount of fragmentation.
      #   (2) A match of one edge with the minimum area left over after the split.
      #   (3) No edges match, so look for the node which leaves the least amount of area left over after the split.
      t = self.textures[index]

      leastY = 0x7FFFFFFF
      leastX = 0x7FFFFFFF

      previousBestFit = None
      bestFit = None
      previous = None
      search = self.free
      edgeCount = 0

      # Walk the singly linked list of free nodes
      # see if it will fit into any currently free space
      while search:
        fits, ec = search.fits(t.width,t.height) # see if the texture will fit into this slot, and if so how many edges does it share.
        if fits:
          if ec == 2:
            previousBestFit = previous     # record the pointer previous to this one (used to patch the linked list)
            bestFit = search # record the best fit.
            edgeCount = ec
            break
          if search.y < leastY:
            leastY = search.y
            leastX = search.x
            previousBestFit = previous
            bestFit = search
            edgeCount = ec
          elif search.y == leastY and search.x < leastX:
            leastX = search.x
            previousBestFit = previous
            bestFit = search
            edgeCount = ec
        previous = search
        search = search.next

      # if bestFit is None:
      #   import pdb; pdb.set_trace()

      # assert bestFit is not None # we should always find a fit location!

      if bestFit:
        self.validate()

        if edgeCount == 0:
          if t.longest_edge <= bestFit.width:
            flipped = False

            wid = t.width
            hit = t.height

            if hit > wid:
              wid = t.height
              hit = t.width
              flipped = True

            t.place(bestFit.x,bestFit.y,flipped) # place it.

            self.new_free(bestFit.x,bestFit.y+hit,bestFit.width,bestFit.height-hit)

            bestFit.x+=wid
            bestFit.width-=wid
            bestFit.height = hit
            self.validate()
          else:

            assert( t.longest_edge <= bestFit.height )

            flipped = False

            wid = t.width
            hit = t.height

            if hit < wid:
              wid = t.height
              hit = t.width
              flipped = True

            t.place(bestFit.x,bestFit.y,flipped) # place it.

            self.new_free(bestFit.x,bestFit.y+hit,bestFit.width,bestFit.height-hit)

            bestFit.x+=wid
            bestFit.width-=wid
            bestFit.height = hit
            self.validate()
        elif edgeCount == 1:
          if t.width == bestFit.width:
            t.place(bestFit.x,bestFit.y,False)
            bestFit.y+=t.height
            bestFit.height-=t.height
            self.validate()
          elif t.height == bestFit.height:
            t.place(bestFit.x,bestFit.y,False)
            bestFit.x+=t.width
            bestFit.width-=t.width
            self.validate()
          elif t.width == bestFit.height:
            t.place(bestFit.x,bestFit.y,True)
            bestFit.x+=t.height
            bestFit.width-=t.height
            self.validate()
          elif t.height == bestFit.width:
            t.place(bestFit.x,bestFit.y,True)
            bestFit.y+=t.width
            bestFit.height-=t.width
            self.validate()
        elif edgeCount == 2:
          flipped = t.width != bestFit.width or t.height != bestFit.height
          t.place(bestFit.x,bestFit.y,flipped)
          if previousBestFit:
            previousBestFit.next = bestFit.next
          else:
            self.free = bestFit.next
          #delete bestFit
          self.validate()
        else:
          import pdb;
          pdb.set_trace()
          assert False

        while self.merge_nodes(): # keep merging nodes as much as we can...
          pass

      
    width = 0
    height = 0
    for t in self.textures:
      if self.one_pixel_border:
        t.width-=2
        t.height-=2
        t.x+=1
        t.y+=1
      if t.placed:
        x = t.x+t.width
        y = t.y+t.height
        if x > width:
          width = x
        if y > height:
          height = y

    if self.force_power_of_two:
      height = nextPow2(height)
    else:
      width = ((width+3)//4)*4
      height = ((height+3)//4)*4

    return (width*height)-self.calc_total_area(placed=True), width, height
    #return width, height, count

  def calc_total_area(self, placed):
    return sum([tex.area for tex in self.textures if tex.placed == placed])

  def calc_longest_edge(self):
    return max(self.textures, key=lambda tex: tex.longest_edge).longest_edge

  def merge_nodes(self):
    f = self.free
    while f:
      prev = None;
      c = self.free
      while c:
        if f is not c:
          if f.merge(c):
            assert prev
            prev.next = c.next
            return True
        prev = c
        c = c.next
      f = f.next
    return False;

  def validate(self):
    return
    f = self.free
    while f:
      c = self.free
      while c:
        if f is not c:
          f.validate(c)
        c = c.next
      f = f.next

  def get_texture_locations(self):
    r = []
    for t in self.textures:
      x = t.x;
      y = t.y;
      if t.flipped:
        wid = t.height;
        hit = t.width;
      else:
        wid = t.width;
        hit = t.height;
      ret = t.flipped;
      r.append(Texture(wid,hit,x,y,flipped=t.flipped,placed=t.placed))
    return r
  

# tp = texture_packer.TexturePacker(one_pixel_border=True, force_power_of_two=False, width=4096, height=4096); images = [tft.bytes_to_pil(x) for x in images4]; tp.textures.extend([texture_packer.Texture(*x.size) for x in images]); tp.pack_textures(); area, w, h = _; w, h; image = Image.new('RGBA', (w, h)); draw = ImageDraw.Draw(image); pp([x if x.placed else None for x in tp.get_texture_locations()])
# [image.alpha_composite((img.transpose(Image.TRANSPOSE) if tex.flipped else img).convert('RGBA'), (tex.x, tex.y)) for tex, img in zip(tp.get_texture_locations(), images) if tex.placed];
# image.convert('RGB').save('test.jpg', quality=60)

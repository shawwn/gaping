import numpy as np

# https://arxiv.org/pdf/2012.11551.pdf

def randn():
  #return (np.random.uniform()*2-1)*3.1415926/2
  return np.random.normal()

def f(z1, z2, epsilon):
  return 3*z1 + 0.1*epsilon, np.cos(3*z1) + np.tanh(3 * z2) + 0.1*epsilon

def points(randomness=0.05):
  for t in np.linspace(0.0, 3.14, 300):
    x, y = f(t, 1.0*t, randomness*randn())
    yield x, y

def randpoints(n):
  for i in range(n):
    x, y = f(randn(), randn(), randn())
    yield x, y

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  plt.figure(figsize=(9, 3))
  xs = []
  ys = []
  for x, y in randpoints(5000):
    xs += [x]
    ys += [y]
  plt.scatter(xs, ys, s=0.1, alpha=0.5)
  plt.show()



import gaping.tftorch as nn
import gaping.tftorch as F
import gaping.models.inception

# Module that wraps the inception network to enable use with dataparallel and
# returning pool features and logits.
class WrapInception(nn.Module):
  def __init__(self, net):
    super(WrapInception,self).__init__()
    self.net = net
    self.mean = nn.view([0.485, 0.456, 0.406], 1, 1, 1, -1)
    self.std = nn.view([0.229, 0.224, 0.225], 1, 1, 1, -1)
  def forward(self, x):
    # Normalize x
    x = (x + 1.) / 2.0
    x = (x - self.mean) / self.std
    # Upsample if necessary
    #if x.shape[2] != 299 or x.shape[3] != 299:
    #  x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)
    x = F.interpolate(x, size=(299, 299), mode='area', align_corners=True)
    # 299 x 299 x 3
    x = self.net.Conv2d_1a_3x3(x)
    # 149 x 149 x 32
    x = self.net.Conv2d_2a_3x3(x)
    # 147 x 147 x 32
    x = self.net.Conv2d_2b_3x3(x)
    # 147 x 147 x 64
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    # 73 x 73 x 64
    x = self.net.Conv2d_3b_1x1(x)
    # 73 x 73 x 80
    x = self.net.Conv2d_4a_3x3(x)
    # 71 x 71 x 192
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    # 35 x 35 x 192
    x = self.net.Mixed_5b(x)
    # 35 x 35 x 256
    x = self.net.Mixed_5c(x)
    # 35 x 35 x 288
    x = self.net.Mixed_5d(x)
    # 35 x 35 x 288
    x = self.net.Mixed_6a(x)
    # 17 x 17 x 768
    x = self.net.Mixed_6b(x)
    # 17 x 17 x 768
    x = self.net.Mixed_6c(x)
    # 17 x 17 x 768
    x = self.net.Mixed_6d(x)
    # 17 x 17 x 768
    x = self.net.Mixed_6e(x)
    # 17 x 17 x 768
    # 17 x 17 x 768
    x = self.net.Mixed_7a(x)
    # 8 x 8 x 1280
    x = self.net.Mixed_7b(x)
    # 8 x 8 x 2048
    x = self.net.Mixed_7c(x)
    # 8 x 8 x 2048
    #pool = torch.mean(x.view(x.size(0), x.size(1), -1), 2)
    pool = self.net.avgpool(x)
    # 1 x 1 x 2048
    pool = nn.flatten(pool, 1)
    # 2048
    #logits = self.net.fc(F.dropout(pool, training=False).view(pool.size(0), -1))
    logits = self.net.fc(pool)
    # 1000 (num_classes)
    return pool, logits

# A pytorch implementation of cov, from Modar M. Alfadly
# https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/2
def torch_cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if nn.dim(m) > 2:
        raise ValueError('m has more than 2 dimensions')
    if nn.dim(m) < 2:
        m = nn.view(m, 1, -1)
    # if not rowvar and nn.size(m, 0) != 1:
    #     m = nn.t(m)
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (nn.size(m, 1) - 1)
    m -= nn.mean(m, dim=1, keepdim=True)
    mt = nn.t(m)  # if complex: mt = m.t().conj()
    return fact * nn.squeeze(nn.matmul(m, mt))
  


# FID calculator from TTUR--consider replacing this with GPU-accelerated cov
# calculations using torch?
def numpy_calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
  """Numpy implementation of the Frechet Distance.
  Taken from https://github.com/bioinf-jku/TTUR
  The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
  and X_2 ~ N(mu_2, C_2) is
          d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
  Stable version by Dougal J. Sutherland.
  Params:
  -- mu1   : Numpy array containing the activations of a layer of the
             inception net (like returned by the function 'get_predictions')
             for generated samples.
  -- mu2   : The sample mean over activations, precalculated on an 
             representive data set.
  -- sigma1: The covariance matrix over activations for generated samples.
  -- sigma2: The covariance matrix over activations, precalculated on an 
             representive data set.
  Returns:
  --   : The Frechet Distance.
  """

  mu1 = np.atleast_1d(mu1)
  mu2 = np.atleast_1d(mu2)

  sigma1 = np.atleast_2d(sigma1)
  sigma2 = np.atleast_2d(sigma2)

  assert mu1.shape == mu2.shape, \
    'Training and test mean vectors have different lengths'
  assert sigma1.shape == sigma2.shape, \
    'Training and test covariances have different dimensions'

  diff = mu1 - mu2

  # Product might be almost singular
  covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
  if not np.isfinite(covmean).all():
    msg = ('fid calculation produces singular product; '
           'adding %s to diagonal of cov estimates') % eps
    print(msg)
    offset = np.eye(sigma1.shape[0]) * eps
    covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

  # Numerical error might give slight imaginary component
  if np.iscomplexobj(covmean):
    #print('wat')
    if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
      m = np.max(np.abs(covmean.imag))
      raise ValueError('Imaginary component {}'.format(m))
    covmean = covmean.real  

  tr_covmean = np.trace(covmean) 

  out = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
  return out


import tensorflow.compat.v1 as tf
import numpy as np
from scipy import linalg # For numpy FID

# def calculate_fid(model, pool1, pool2, session=None):
#   session = session or tf.get_default_session()
#   mu1, sigma1 = nn.mean(pool1, 0), torch_cov(pool1, rowvar=False)
#   mu2, sigma2 = nn.mean(pool2, 0), torch_cov(pool2, rowvar=False)
#   mu1, sigma1, mu2, sigma2 = session.run([mu1, sigma1, mu2, sigma2])
#   return numpy_calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

def numpy_calculate_fid(model, pool1, pool2):
  mu1, sigma1 = np.mean(pool1, 0), np.cov(pool1)
  mu2, sigma2 = np.mean(pool2, 0), np.cov(pool2)
  return numpy_calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

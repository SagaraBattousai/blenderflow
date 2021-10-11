import math

def ctor(x, y):
  r = ((x ** 2) + (y ** 2)) ** 0.5
  theta = 0
  if x == 0 and y != 0:
    theta = math.pi / 2 if y > 0 else (math.pi / -2)
  else:
    theta = math.atan(y / x)

  return r, theta

def rtoc(r, theta):
  x = r * math.cos(theta)
  y = r * math.sin(theta)
  return x, y

def rotate(theta, x, y):
  xp = x * math.cos(theta) - y * math.sin(theta)
  yp = x * math.sin(theta) + y * math.cos(theta)
  return xp, yp
  

if __name__ == "__main__":

  import matplotlib.pyplot as plt

  theta = [t * (math.pi / 180) for t in range(0, 361)]
  # r = [(4 * math.cos(t) ** 2 + math.sin(t) ** 2) ** 0.5 for t in theta]

  # xy = [rtoc(ri, ti) for ri, ti in zip(r, theta)]

  # xs, ys = list(zip(*xy))

  # xryr = [rotate((math.pi / 4), x, y) for x, y in xy]

  # xr, yr = list(zip(*xryr))

  
  # plt.plot(xs, ys)
  # plt.plot(xr, yr)
  # plt.show()

  ys = [math.sin(t) for t in theta]
  rotsin = [rotate((math.pi / 4), t, y) for t, y in zip(theta, ys)]
  rotx, roty = list(zip(*rotsin))

  def xrotf(t, x):
    return x * math.cos(t) - math.sin(t)*math.sin(x)

  rt = math.pi / 4

  yidea = [xrotf(rt, t) * math.sin(rt) + math.cos(rt) * math.sin(xrotf(rt, t)) for t in theta]
  

  plt.plot(theta, ys)
  plt.plot(rotx, roty)
  plt.plot(theta, roty)
  plt.plot(theta, yidea)
  plt.plot([1,2,3,4,5,6])
  plt.show()
  



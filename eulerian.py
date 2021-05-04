import taichi as ti
import numpy as np
import time

#################################
## Initialization and Settings ##
#################################

ti.init(arch=ti.gpu)

grid_size = (200, 200)  # size of grids
dx = 3  # size of each grid
tex_size = tuple([dx * i for i in grid_size])  # size of texture
diff = 1.0  # mu in diffusion equation
force_rad = grid_size[0] / 4  # radius in computation of additional forces

u_old = ti.Vector(2, ti.f32)  # fluid velocity
u_new = ti.Vector(2, ti.f32)
p_old = ti.Vector(1, ti.f32)
p_new = ti.Vector(1, ti.f32)
div = ti.var(ti.f32)
ti.root.dense(ti.ij, grid_size).place(u_old, u_new, p_old, p_new, div)

tex = ti.Vector(3, ti.f32)  # rendered texture
ti.root.dense(ti.ij, tex_size).place(tex)

force_coord = ti.field(dtype=float, shape=2)  # coordinate of added force
force_add = ti.field(dtype=float, shape=2)  # added force
is_used = False  # if already added

@ti.kernel
def init():
    for i, j in u_old:
        u_old[i, j].x = (i + j) % 30
        u_old[i, j].y = (i - j) % 30

####################
## Help functions ##
####################

# Jacobi loop for discrete poisson (without boundary condition)
@ti.func
def jacobiOp(x, b, coord, alpha, beta):
    x_neighb = x[coord + ti.Vector([1, 0], ti.int32)] \
               + x[coord + ti.Vector([-1, 0], ti.int32)] \
               + x[coord + ti.Vector([0, 1], ti.int32)] \
               + x[coord + ti.Vector([0, -1], ti.int32)]
    x_new = (x_neighb + b[coord] * alpha) * beta
    return x_new

# boundary operation
@ti.func
def boundOp(x, coord, offset, param):
    x[coord] = x[coord + offset] * param

# copy from x to y
@ti.kernel
def copy(x: ti.template(), y: ti.template()):
    for I in ti.grouped(x):
        y[I] = x[I]

# clear vectors
@ti.kernel
def clear(x: ti.template()):
    for I in ti.grouped(x):
        x[I].fill(0)

# exponential kernel
@ti.kernel
def expKernel(x1: ti.template(), x2:ti.template()):
  return ti.exp(-(x1-x2).norm() / force_rad)

# gui event process
# def eventProc():
#   gui.get_event()
#   if gui.is_pressed(ti.GUI.LMB, type=ti.GUI.PRESS):
#     force_coord[0], force_coord[1] = gui.get_cursor_pos()

# bilinear interpolation function: x -> coord, y -> val mat
@ti.func
def bilinearInterpolation(x, y):
    x_floor = ti.floor(x)
    x_dec = x - x_floor
    y_new = y[int(x_floor)] * (1. - x_dec[0]) * (1. - x_dec[1]) \
            + y[int(x_floor) + ti.Vector([1, 0], ti.int32)] * x_dec[0] * (1. - x_dec[1]) \
            + y[int(x_floor) + ti.Vector([0, 1], ti.int32)] * (1. - x_dec[0]) * x_dec[1] \
            + y[int(x_floor) + 1] * x_dec[0] * x_dec[1]
    return y_new

###############
## Advection ##
###############

def advection(dt):
    advectionOp(dt)  # computation
    copy(u_new, u_old)  # copy data

@ti.kernel
def advectionOp(dt: ti.f32):
    for I in ti.grouped(u_old):
        pos_old = I - dt * u_old[I] / dx
        pos_old = min(max(pos_old, 0), ti.static(grid_size))
        u_new[I] = bilinearInterpolation(pos_old, u_old)

###############
## Diffusion ##
###############

def diffusion(dt, iter):
  clear(u_new)
  a = 1. / (diff * dt)
  b = 1. / (4 + a)
  for i in range(iter):
    diffusionLoop(a, b)
    copy(u_new, u_old)

@ti.kernel
def diffusionLoop(alpha: ti.f32, beta: ti.f32):
    # jacobi op
    for I in ti.grouped(u_new):
        if I[0] * I[1] > 0 and I[0] < grid_size[0] - 1 and I[1] < grid_size[1] - 1:
            u_new[I] = jacobiOp(u_new, u_old, I, alpha, beta)
    # bnd op
    for I in ti.grouped(u_new):
        offset = [0, 0]
        offset[0] = 1 if I[0] == 0 else 0
        offset[0] = -1 if I[0] == grid_size[0] - 1 else 0
        offset[1] = 1 if I[1] == 0 else 0
        offset[1] = -1 if I[1] == grid_size[1] - 1 else 0
        boundOp(u_new, I, ti.Vector(offset, ti.int32), -1.0)

#####################
## External Forces ##
#####################

# @ti.kernel
# def addForceOp(dt:ti.f32, F:ti.template(), coord:ti.Vector(2, ti.int32)):
#   for I in ti.grouped(u_old):
#     u_old[I] += F * dt * expKernel(I, coord)

@ti.kernel
def addRandomForce(dt:ti.f32):
    pos_p = ti.Vector([ti.random(), ti.random()])
    F = ti.Vector([ti.random(), ti.random()]) * ti.random() * 1000.0
    # if ti.random() > 0.01:
    #     F.fill(0)
    for I in ti.grouped(u_old):
        u_old[I] += F * dt * ti.exp(-(I-pos_p).norm()/0.1)

################
## Projection ##
################

@ti.kernel
def computeUDiv():
    for I in ti.grouped(u_old):
        if I.norm() > 0 and I[0] < grid_size[0] and I[1] < grid_size[1]:
            dif1 = u_old[I + ti.Vector([1, 0])] - u_old[I - ti.Vector([1, 0])]
            dif2 = u_old[I + ti.Vector([0, 1])] - u_old[I - ti.Vector([0, 1])]
            div[I] = (dif1.x + dif2.y) * 0.5

@ti.kernel
def pressureLoop(dt:ti.f32):
    # Jacobi loop
    for I in ti.grouped(u_new):
        if I[0] * I[1] > 0 and I[0] < grid_size[0] - 1 and I[1] < grid_size[1] - 1:
            p_new[I] = jacobiOp(p_old, div, I, -1.0, 0.25)
    # bnd op
    for I in ti.grouped(p_new):
        offset = [0, 0]
        offset[0] = 1 if I[0] == 0 else 0
        offset[0] = -1 if I[0] == grid_size[0] - 1 else 0
        offset[1] = 1 if I[1] == 0 else 0
        offset[1] = -1 if I[1] == grid_size[1] - 1 else 0
        boundOp(p_new, I, ti.Vector(offset, ti.int32), 1.0)

@ti.kernel
def gradientSubtraction():
    # div compute
    for I in ti.grouped(div):
        if I[0] * I[1] > 0 and I[0] < grid_size[0] - 1 and I[1] < grid_size[1] - 1:
            dif1 = p_old[I + ti.Vector([1, 0])] - p_old[I - ti.Vector([1, 0])]
            dif2 = p_old[I + ti.Vector([0, 1])] - p_old[I - ti.Vector([0, 1])]
            u_new[I] = u_old[I]
            u_new[I].x -= dif1.x * 0.5
            u_new[I].y -= dif2.x * 0.5
    # bnd op
    for I in ti.grouped(u_new):
        offset = [0, 0]
        offset[0] = 1 if I[0] == 0 else 0
        offset[0] = -1 if I[0] == grid_size[0] - 1 else 0
        offset[1] = 1 if I[1] == 0 else 0
        offset[1] = -1 if I[1] == grid_size[1] - 1 else 0
        boundOp(u_new, I, ti.Vector(offset, ti.int32), -1.0)


def projection(dt, iter):
    computeUDiv()
    clear(p_old)
    for i in range(iter):
        pressureLoop(dt)
        copy(p_new, p_old)
    gradientSubtraction()


#####################
## Step and Render ##
#####################

# render from x to texture
@ti.kernel
def render(x: ti.template()):
    p_max = 0.0
    p_min = 0.0
    for I in ti.grouped(tex):
      #tex[I].fill(x[I // dx].norm())

      # tex[I][0] = ti.abs(x[I // dx][0])/x[I // dx].norm()
      # tex[I][1] = ti.abs(x[I // dx][1])/x[I // dx].norm()
      # tex[I][2] = 1.0
      p_max = p_old[I].x if p_old[I].x > p_max else p_max
      p_min = p_old[I].x if p_old[I].x < p_min else p_min
      # x += p_old[I].x
    for I in ti.grouped(tex):
      tex[I].fill((p_old[I].x-p_min)/(p_max - p_min))

def step(dt):
    advection(dt)
    diffusion(dt, 40)
    #addRandomForce(dt)
    projection(dt, 100)

####################
## Main Execution ##
####################

init()
gui = ti.GUI("Test", tex_size, background_color=0x112F41, fast_gui=True)
for i in range(180):
#while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
    step(1. / 100)
    render(u_old)
    #gui.set_image(tex)
    if (i==49):
        ti.imwrite(tex.to_numpy(), 'test.png')
    #gui.show(f'test.png')

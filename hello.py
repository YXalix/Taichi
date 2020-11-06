import taichi as ti

ti.init(arch = ti.cuda)
a = ti.field(ti.f32,(42,63))
b = ti.Vector(3,dt = ti.f32,shape=4)
c = ti.Matrix(2,2,dt = ti.f32,shape = (3,5))

loss = ti.field(ti.f32,shape=())

a[3,4] = 1
print(a[3,4])
print(a)
b[0] = [6,7,8]

print(b[0][0],b[0][1],b[0][2])

loss[None] = 3
print(loss[None])

@ti.kernel

def hello(i:ti.i32):
    a = 40
    print('hello world',a+i)

hello(2)


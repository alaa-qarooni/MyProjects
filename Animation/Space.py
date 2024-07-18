
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

import pymunk
from pymunk.vec2d import Vec2d

def setup_space(width,height,e):
    space = pymunk.Space()
    space.gravity = 0,0
    space.damping = 0.9999
    static_body = space.static_body
    gap = 0.1
    static_lines = [
        #Top Ceiling
        pymunk.Segment(static_body, (gap, height - gap), (width-gap, height - gap), 0.01),
        #Bottom Floor
        pymunk.Segment(static_body, (gap,gap), (width-gap, gap), 0.01),
        # Right wall
        pymunk.Segment(static_body, (width - gap, gap), (width - gap, height - gap), 0.01),
        # Left wall
        pymunk.Segment(static_body, (gap, gap), (gap, height - gap), 0.01)
    ]
    for line in static_lines:
        line.elasticity = e
        line.friction = 0
    space.add(*static_lines)
    return space

def mk_ball(x,y,vx,vy,radius,e,space):
    body = pymunk.Body(0,0)
    body.position = Vec2d(x, y)
    body.velocity = Vec2d(vx, vy)
    #body.start_position = Vec2d(*body.position)
    shape = pymunk.Circle(body, radius)
    shape.density = 1
    shape.elasticity = e
    space.add(body, shape)
    body.radius = radius
    return body

def initialize():
    width , height = 16,9
    e = 0.95 # elasticity of pbjects
    space = setup_space(width, height, e)

    #create N balls with radius r
    N, r = 40, 0.2

    # velocity of each all in the tangential directioon
    vt = 3.0
    # random component of each ball's velocity (uniform)
    vrand = 1.0
    
    balls = []
    for tx,ty in zip(np.random.uniform(0,width-1,N),np.random.uniform(0,height-1,N)):
        balls.append(mk_ball(
            x = tx,
            y = ty,
            vx = vt*np.random.uniform(-vrand,+vrand),
            vy = vt*np.random.uniform(-vrand,+vrand),
            radius = r,
            e = e,
            space = space
        ))
    
    # create our kinematic object
    our_guy = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    # Randomize starting position
    our_guy.position = (np.random.randint(2,width-2),np.random.randint(2,height-2))
    our_guy.velocity = 0,0
    our_guy.radius = 0.5
    shape=pymunk.Circle(our_guy,0.5)
    shape.elasticity=0.7
    space.add(our_guy,shape)

    for b in balls:
        if Vec2d.get_distance(our_guy.position,b.position)<1:
            b.position = (0.5,0.5)

    return width, height, space
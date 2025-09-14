
import numpy as np

import pymunk
from pymunk.vec2d import Vec2d

def setup_space(width,height,e):
    space = pymunk.Space()
    space.gravity = 0,0
    space.damping = 0.9999
    static_body = space.static_body
    gap = 5
    static_lines = [
        #Top Ceiling
        pymunk.Segment(static_body, (gap, height - gap), (width-gap, height - gap), 10),
        #Bottom Floor
        pymunk.Segment(static_body, (gap,gap), (width-gap, gap), 10),
        # Right wall
        pymunk.Segment(static_body, (width - gap, gap), (width - gap, height - gap), 10),
        # Left wall
        pymunk.Segment(static_body, (gap, gap), (gap, height - gap), 10)
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

def mk_our_guy(width,height,space,rays=12):
    # create our kinematic object
    our_guy = pymunk.Body(0,moment=float('inf'))
    # Randomize starting position
    our_guy.position = (np.random.randint(100,1000),np.random.randint(100,600))
    our_guy.velocity = Vec2d(0,0)

    # Main Circle
    shape = pymunk.Circle(our_guy,50)
    shape.density=1
    shape.elasticity=0.7
    shape.collision_type = 1

    sensors=12
    shapes = []
    # for i in range(sensors):
    #     start_pos = shape.radius*Vec2d(np.cos(i*2*np.pi/sensors),np.sin(i*2*np.pi/sensors))
    #     end_pos = start_pos + 100*Vec2d(np.cos(i*2*np.pi/sensors),np.sin(i*2*np.pi/sensors))
        
    #     # Perform the raycast
    #     shapes.append(pymunk.Segment(our_guy,start_pos,end_pos,radius=1))
    #     shapes[-1].sensor=True
    #     shapes[-1].collision_type=1

    space.add(our_guy,shape,*shapes)
    return our_guy

def initialize(width,height,n_balls, ball_size, ball_speed):
    e = 0.95 # elasticity of pbjects
    space = setup_space(width, height, e)

    #create N balls with radius r
    N, r = n_balls, ball_size

    # velocity of each ball in the tangential direction
    vt = ball_speed
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
    
    our_guy = mk_our_guy(width,height,space)

    for b in balls:
        if Vec2d.get_distance(our_guy.position,b.position)<100:
            b.position = (r+0.5,r+0.5)

    return width, height, space
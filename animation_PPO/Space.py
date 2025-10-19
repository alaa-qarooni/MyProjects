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
        pymunk.Segment(static_body, (gap, height - gap), (width-gap, height - gap), 10),
        pymunk.Segment(static_body, (gap,gap), (width-gap, gap), 10),
        pymunk.Segment(static_body, (width - gap, gap), (width - gap, height - gap), 10),
        pymunk.Segment(static_body, (gap, gap), (gap, height - gap), 10)
    ]
    for line in static_lines:
        line.elasticity = e
        line.friction = 0
    space.add(*static_lines)
    return space

def mk_ball(x,y,vx,vy,radius,e,space, color=(1.0, 0.0, 0.0, 1.0)):  # RGBA format, 0-1.0 range
    body = pymunk.Body(0,0)
    body.position = Vec2d(x, y)
    body.velocity = Vec2d(vx, vy)
    shape = pymunk.Circle(body, radius)
    shape.density = 1
    shape.elasticity = e
    shape.color = color
    shape.collision_type = 1
    space.add(body, shape)
    return body, shape

def mk_target(width, height, space, radius=30):
    """Create a target ball that the agent needs to reach, ensuring it spawns in a clear area"""
    target = pymunk.Body(body_type=pymunk.Body.STATIC)
    
    # Try to find a clear position (up to 50 attempts)
    for _ in range(50):
        pos = (np.random.randint(100, width-100), np.random.randint(100, height-100))
        
        # Check if this position has clear space around it
        clear_space = True
        
        # Create a temporary shape to test for collisions
        test_shape = pymunk.Circle(None, radius + 50)  # 100x100 area (radius 50 from center)
        test_shape.body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        test_shape.body.position = pos
        
        # Query for shapes in the area
        query = space.shape_query(test_shape)
        
        # Check if any non-target shapes are in the area
        for shape in query:
            if (hasattr(shape, 'collision_type') and 
                shape.collision_type != 2 and  # Not a target
                shape.body.body_type != pymunk.Body.STATIC):  # Not a wall
                clear_space = False
                break
        
        if clear_space:
            target.position = pos
            break
    else:
        # If no clear space found after 50 attempts, use random position
        target.position = (np.random.randint(100, width-100), np.random.randint(100, height-100))
    
    shape = pymunk.Circle(target, radius)
    shape.density = 0
    shape.elasticity = 0
    shape.sensor = True
    shape.color = (0.0, 1.0, 0.0, 1.0)  # Green in RGBA
    shape.collision_type = 2
    space.add(target, shape)
    return target, shape

def mk_our_guy(width,height,space,radius=50):
    # create our kinematic object
    our_guy = pymunk.Body(0,moment=float('inf'))
    our_guy.position = (np.random.randint(100,1000),np.random.randint(100,600))
    our_guy.velocity = Vec2d(0,0)

    # Main Circle
    shape = pymunk.Circle(our_guy,radius)
    shape.density=1
    shape.elasticity=0.7
    shape.collision_type = 1
    shape.color = (0.0, 0.0, 1.0, 1.0)  # Blue in RGBA

    space.add(our_guy,shape)
    return our_guy, shape

def get_body_radius(body):
    """Helper function to get radius from body's shape"""
    for shape in body.shapes:
        if isinstance(shape, pymunk.Circle):
            return shape.radius
    return 0

def initialize(width,height,n_balls, ball_size, ball_speed):
    e = 0.95
    space = setup_space(width, height, e)

    # Create target ball
    target_body, target_shape = mk_target(width, height, space)

    # Create N balls with radius r
    N, r = n_balls, ball_size
    vt = ball_speed
    vrand = 1.0
    
    balls = []
    for tx,ty in zip(np.random.uniform(0,width-1,N),np.random.uniform(0,height-1,N)):
        ball_body, _ = mk_ball(
            x = tx,
            y = ty,
            vx = vt*np.random.uniform(-vrand,+vrand),
            vy = vt*np.random.uniform(-vrand,+vrand),
            radius = r,
            e = e,
            space = space
        )
        balls.append(ball_body)
    
    our_guy_body, our_guy_shape = mk_our_guy(width,height,space)

    # Ensure balls don't spawn too close to our guy or target
    for b in balls:
        if Vec2d.get_distance(our_guy_body.position,b.position)<150:
            b.position = (r+0.5,r+0.5)
        if Vec2d.get_distance(target_body.position,b.position)<150:
            b.position = (width-r-0.5,height-r-0.5)

    return width, height, space, target_body, target_shape, our_guy_body, our_guy_shape
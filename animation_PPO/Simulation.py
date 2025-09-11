import Space
import NN

import numpy as np
import math
import torch
import itertools
import pickle
from tqdm import tqdm
import random

import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import pygame
import glom

PYTORCH_ENABLE_MPS_FALLBACK=1

def train_model(dt, n_episodes=5000,episode_length=50):
    
    # state is composed of the positions and velocities of 50 dynamic balls,
    # 1 kinematic ball, and 4 static walls.
    # The kinematic balls will take the action from the target network.
    # Action space is velocity and direction kinematic ball can take: 0 rad - 2pi rad
    N = 31
    action_space = list(itertools.product(np.arange(0,4,0.5).tolist(),np.arange(-np.pi,np.pi,np.pi/8).tolist()))
    state_space = [0]*(N+1)*4
    model = NN.simulator(state_space,action_space)


    for ep in range(n_episodes):
        width,height,space = Space.initialize(n_balls=random.choice([10,20,30]), ball_size=random.randrange(0.2,0.6,0.1))
        r = torch.tensor([0.])
        
        bodies = list(space.bodies)[:-1]
        our_guy = list(space.bodies)[-1]

        s = get_state(our_guy, bodies)
        s = torch.cat((torch.tensor([0,0,width,height]).to("mps"),s))

        model.episode=ep

        for t in tqdm(np.arange(0,episode_length,dt)):

            # select action
            a = model.select_action(s, decay=n_episodes/3)

            # Apply action to velocity
            a_vel, a_ang = model.actions[a]
            x_vel = a_vel*math.cos(a_ang)
            y_vel = a_vel*math.sin(a_ang)
            our_guy.velocity = x_vel,y_vel
            
            # Step in simulation
            space.step(dt)

            # Reward for staying in simulation
            r += torch.tensor([dt])
            
            collision = [our_guy.shapes[0].shapes_collide(b.shapes[0]) for b in bodies]
            collision = any(round(c.points[0],2).distance>0 for c in collision)

            # End simulation when collision occurs with penalty
            if collision:
                r-=torch.tensor([50*dt])
                s_prime=None
            else:
            # Get next state
                s_prime = get_state(space.bodies)
                s_prime = torch.cat((torch.tensor([0,0,width,height]).to("mps"),s_prime))

            # Store transition in memory
            model.memory.push(s,a,s_prime,r)

            # Reset reward
            r = torch.tensor([0.])

            # Move to the next state
            s = s_prime

            # Perform one step of optimization
            model.optimize_model()
            
            # Soft update of target weights
            target_net_state_dict = model.target_net.state_dict()
            policy_net_state_dict = model.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*model.TAU + target_net_state_dict[key]*(1-model.TAU)
            model.target_net.load_state_dict(target_net_state_dict)

            if s is None:
                break

    return model

# Collision handling
def begin_col(arbiter, space, data):
    data["col"]=1
    data["tot_col"]+=1
    return True
    
# State is defined by the normalized position of the ball in space and its ray vision
def get_state(space):
    our_guy = list(space.bodies)[-1]
    bodies = list(space.bodies)[:-1]

    

    return torch.tensor(s, dtype=torch.float32).to(NN.device)

def get_ray_trace(space, sensors=12,sensor_range=150):
    our_guy = list(space.bodies)[-1]
    vals = [0]*sensors
    contact_points = []
    for i in range(sensors):
        start_pos = our_guy.position
        end_pos = start_pos + 150*Vec2d(np.cos(i*2*np.pi/sensors),np.sin(i*2*np.pi/sensors))
        
        # Perform the raycast
        hit = space.segment_query(start_pos, end_pos, 0, pymunk.ShapeFilter())
        if len(hit)>1:
            hit = hit[1]
            vals[i] = 1 - hit.alpha
            contact_points.append(hit)
    
    return vals, contact_points

def simulate(width, height, space, velocity_control=0):
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    draw_options = pymunk.pygame_util.DrawOptions(screen)

    running = True
    count=0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        our_guy = list(space.bodies)[-1]
        if not velocity_control:
            our_guy.velocity = Vec2d(0,100)
            count+=1
        else:
            our_guy.velocity = Vec2d(velocity_control[0],velocity_control[1])

        screen.fill((255,255,255))
        
        space.debug_draw(draw_options)

        vals, contact_points = get_ray_trace(space)
        
        for contact in contact_points:
            pygame.draw.circle(screen, (255,0,0), contact.point, 5)

        rays = pygame.font.SysFont("Consolas", 16).render(f"Position: {'%.0f, %.0f' % our_guy.position}", True, (0, 0, 0))
        screen.blit(rays, (10, 10))
        position = pygame.font.SysFont("Consolas", 16).render(f"Sensors: {vals}", True, (0, 0, 0))
        screen.blit(position, (10,30))

        #Step the simulation every 3 frames
        space.step(dt)

        pygame.display.flip()
        clock.tick(1/dt)

    pygame.quit()

if __name__ == "__main__":

    run_name = "first"
    dt = 1/60

    # Train and save the target network, and set of actions to choose from
    # location = "./"
    # if not run_name:
    #     simulator = train_model(dt)
        
    #     network = torch.jit.script(simulator.target_net)
    #     network.save(location+"models/"+run_name+".pt")

    #     with open(location+"actions", "wb") as path:
    #         pickle.dump(simulator.actions, path)
    
    # # Load the network and set of actions to use in the simulation
    # model = {}
    # model["network"] = torch.jit.load(location+"models/"+run_name+".pt")
    # with open(location+"actions", "rb") as path:
    #     model["actions"] = pickle.load(path)
    
    
    # Initialize and run simulation

    width,height,space = Space.initialize(1280,720,n_balls=20,ball_size=20)
    

    vx = 20
    vy = 20
    simulate(width,height,space,velocity_control=())

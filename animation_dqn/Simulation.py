import Space
import NN

import numpy as np
import os
import torch
import itertools
import pickle
from tqdm import tqdm
import random

import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import pygame

PYTORCH_ENABLE_MPS_FALLBACK=1

def train_model(width, height, dt, n_episodes=6500,episode_length=20, pretrained_weights=None):
    
    action_space = list(
        itertools.product(
            range(-200,201,200),range(-200,201,200)
        )
    )
    state_space = [0]*16
    model = NN.simulator(state_space,action_space,pretrained_weights)
    a = torch.tensor([4],dtype=torch.long,device=NN.device)
    avg_reward = survival_time = 0

    difficulty_schedule = {
        0:      {"n_balls": 5,  "ball_size": 30, "ball_speed": 0},
        1000:   {"n_balls": 10, "ball_size": 25, "ball_speed": 0},
        2500:   {"n_balls": 10, "ball_size": 25, "ball_speed": 50},  # SLOW movement first
        3500:   {"n_balls": 10, "ball_size": 25, "ball_speed": 100}, # Then medium
        4500:   {"n_balls": 15, "ball_size": 20, "ball_speed": 100}, # Then add count
        5500:   {"n_balls": 15, "ball_size": 20, "ball_speed": 200}, # Then increase speed
    }
    
    for ep in range(n_episodes):
        # Get current difficulty based on episode
        current_difficulty = max([k for k in difficulty_schedule.keys() if k <= ep])
        params = difficulty_schedule[current_difficulty]
        
        width, height, space = Space.initialize(width, height, 
                                              n_balls=params["n_balls"], 
                                              ball_size=params["ball_size"],
                                              ball_speed=params["ball_speed"])
        our_guy = list(space.bodies)[-1]
        s,_ = get_state(space)
        model.episode=ep
        steps = 0

        if ep in difficulty_schedule.keys():
            simulate(width,height,space,model.target_net,action_space=action_space,time=20)

        if ep % 100 == 0:
            
            print(f"Ep {ep}: ε={model.eps_thresh:.3f}, "
                f"Loss={model.loss:.4f}, "
                f"Avg Reward={avg_reward/100:.3f}, "
                f"Avg Survival={np.mean(survival_time/100):.1f}s")
            avg_reward=survival_time=0

        for t in np.arange(0,episode_length,dt):
            steps+=1

            # select action
            if steps % steps_per_update == 0:
                a = model.select_action(s, decay=n_episodes/3)
                # Apply action to velocity
                x_vel, y_vel = model.actions[a]
                our_guy.velocity = x_vel,y_vel

            
            # Step in simulation
            space.step(dt)

            # get state
            s_prime, distance_reward = get_state(space)

            # End simulation when collision occurs with penalty
            if not distance_reward:
                r=torch.tensor([-10.])
                s_prime=None
                survival_time += t
            # Reward for staying in simulation
            else:
                if t == episode_length-dt:
                    completion_bonus = 10
                    r = torch.tensor([0.1 + distance_reward + completion_bonus])
                else:
                    r = torch.tensor([0.1 + distance_reward])

            # Store transition in memory
            model.memory.push(s,a,s_prime,r)

            # Move to the next state
            s = s_prime
            avg_reward += r.item()
            
            # Perform one step of optimization
            if steps % steps_per_update == 0:
                model.optimize_model()
            target_update = 10
            # Soft update of target weights
            if steps % (steps_per_update*target_update) == 0:
                target_net_state_dict = model.target_net.state_dict()
                policy_net_state_dict = model.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*model.TAU + target_net_state_dict[key]*(1-model.TAU)
                model.target_net.load_state_dict(target_net_state_dict)

            if s is None:
                break

    return model
    
# State is defined by the normalized position and velocity of the ball in space and its ray vision
def get_state(space):
    our_guy = list(space.bodies)[-1]

    pos = [our_guy.position.x/width, our_guy.position.y/height]
    vel = [our_guy.velocity.x/500, our_guy.velocity.y/500]

    rays, _ = get_ray_trace(space)
    reward = min(rays) * 0.05

    s = pos + vel + rays

    return torch.tensor(s, dtype=torch.float32).to(NN.device), reward

def get_ray_trace(space, sensors=12, sensor_range=200):
    our_guy = list(space.bodies)[-1]
    vals = [1.0]*sensors
    contact_points = []
    for i in range(sensors):
        start_pos = our_guy.position + (list(our_guy.shapes)[0].radius+2)*Vec2d(np.cos(i*2*np.pi/sensors),np.sin(i*2*np.pi/sensors))
        end_pos = start_pos + sensor_range*Vec2d(np.cos(i*2*np.pi/sensors),np.sin(i*2*np.pi/sensors))
        
        # Perform the raycast
        hit = space.segment_query_first(start_pos, end_pos, 0, pymunk.ShapeFilter())
        if hit:
            vals[i] = round(hit.alpha,6)
            if vals[i]==0:
                contact_points.append(start_pos)
            else:
                contact_points.append(hit.point)
    
    return vals, contact_points

def edge_control(width,height,our_guy):
    if round(our_guy.position.x - list(our_guy.shapes)[0].radius) < 15:
        our_guy.velocity = Vec2d(0,our_guy.velocity.y)
    if round(our_guy.position.y - list(our_guy.shapes)[0].radius) < 15:
        our_guy.velocity = Vec2d(our_guy.velocity.x,0)
    if round(our_guy.position.x + list(our_guy.shapes)[0].radius) > width-15:
        our_guy.velocity = Vec2d(0,our_guy.velocity.y)
    if round(our_guy.position.y + list(our_guy.shapes)[0].radius) > height-15:
        our_guy.velocity = Vec2d(our_guy.velocity.x,0)

def simulate(width, height, space, model, action_space, time=float("inf")):
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

        if count*dt>time:
            running=False
        
        our_guy = list(space.bodies)[-1]
        count+=1
        if not action_space:
            our_guy.velocity = Vec2d(500*np.sin(count*2*np.pi/300),0)
        else:
            if count%steps_per_update==0:
                our_guy.velocity = NN.select_action_trained(model,get_state(space)[0],action_space)

        edge_control(width,height,our_guy)

        screen.fill((255,255,255))
        
        space.debug_draw(draw_options)

        vals, contact_points = get_ray_trace(space)
        
        for contact in contact_points:
            pygame.draw.circle(screen, (255,0,0), contact, 5)

        rays = pygame.font.SysFont("Consolas", 16).render(f"Position: {'%.4f, %.4f' % (our_guy.position.x/width, our_guy.position.y/height)}", True, (0, 0, 0))
        screen.blit(rays, (10, 10))
        position = pygame.font.SysFont("Consolas", 16).render(f"Sensors: {vals}", True, (0, 0, 0))
        screen.blit(position, (10,30))
        position = pygame.font.SysFont("Consolas", 16).render(f"{count*dt}s", True, (0, 0, 0))
        screen.blit(position, (width-10,height-10))
        
        space.step(dt)

        pygame.display.flip()
        clock.tick(1/dt)

    pygame.quit()

if __name__ == "__main__":

    dt = 1/60
    steps_per_update = 10

    # Initialize and run simulation
    width,height,space = Space.initialize(1280,720,n_balls=20,ball_size=20, ball_speed = 0)

    # Train and save the target network, and set of actions to choose from
    run_name = "third"
    train = False
    location = "animation_dqn/models/"
    weights_path = location+run_name+".pt"
    if train:
        simulator = train_model(width, height, dt, pretrained_weights=None)
        
        torch.jit.script(simulator.target_net).save(weights_path)
        with open(location+"actions", "wb") as file:
            pickle.dump(simulator.actions, file)
    
    # Load the network and set of actions to use in the simulation
    model = {}
    model["network"] = torch.jit.load(weights_path)
    with open(location+"actions", "rb") as file:
        model["actions"] = pickle.load(file)
    
    simulate(width,height,space,model["network"],model["actions"])
    

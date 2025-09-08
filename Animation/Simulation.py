import Space
import NN

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import wandb

import os
import math
import torch
import itertools
import pickle
from collections import deque
from tqdm import tqdm
import time

import pymunk
from pymunk.vec2d import Vec2d

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
    col=0

    wandb.init(project="ball-simulation", name="")

    model.LR = wandb.config["LR"]
    model.GAMMA = wandb.config["GAMMA"]
    model.TAU = wandb.config["TAU"]
    model.BATCH_SIZE = wandb.config["BATCH_SIZE"]
    model.dropout_rate = wandb.config["DROPOUT"]
    penalty = wandb.config["PENALTY"]
    vicinity = wandb.config["VICINITY"]
    model.n_layers = wandb.config["N_LAYERS"]
    model.dim = wandb.config["DIM"]

    global run_name
    run_name = wandb.run.name


    for ep in range(n_episodes):
        width,height,space = Space.initialize()
        r = torch.tensor([0.])
        
        bodies = space.bodies

        s = get_state(space.bodies)
        s = torch.cat((torch.tensor([0,0,width,height]).to("mps"),s))

        sim_per_s = 4
        model.episode=ep

        ball_col = space.add_collision_handler(0,1)
        ball_col.data["col"] = 0
        ball_col.data["tot_col"] = 0
        ball_col.begin = begin_ball_col

        wall_col = space.add_collision_handler(0,2)
        wall_col.data["col"] = 0
        wall_col.begin = begin_wall_col
        c=0

        for t in tqdm(np.arange(0,episode_length,dt)):

            if c%(1/dt/sim_per_s)==0:
                # select action
                a = model.select_action(s, decay=n_episodes/3)

                # Apply action to velocity
                a_vel, a_ang = model.actions[a]
                x_vel = a_vel*math.cos(a_ang)
                y_vel = a_vel*math.sin(a_ang)
                bodies[-1].velocity = x_vel,y_vel
            
            # Step in simulation
            space.step(dt)

            # Flip velocity if it hits the boundary
            flip_velocity_if_boundary(width,height,bodies[-1], wall_col.data["col"])

            # Reward for staying in simulation
            r += torch.tensor([dt])

            # Penalize as ball approaches the walls
            r -= 0.1*dt*torch.tensor([abs(bodies[-1].position.x - width/2) + abs(bodies[-1].position.y - height/2)])/sim_per_s

            # Penalize when any one ball is within specified vicinity
            if any([torch.linalg.norm(torch.tensor(b.position-bodies[-1].position)) < bodies[-1].radius+vicinity for b in bodies[:-1]]):
                r -= torch.tensor([penalty*dt])
            # Penalize higher when ball is within specified vicinity of wall:
            if bodies[-1].position.x < bodies[-1].radius+vicinity or bodies[-1].position.y < bodies[-1].radius+vicinity or bodies[-1].position.x>width-(bodies[-1].radius+vicinity) or bodies[-1].position.y>height-(bodies[-1].radius+vicinity):
                r -= torch.tensor([2*penalty*dt])
            

            if c%(1/dt/sim_per_s)==0:
                # End simulation when 3 ball collisions or 1 wall collision occur
                if ball_col.data["col"]:
                    r-=torch.tensor([50*dt])
                    s_prime=None
                elif wall_col.data["col"]:
                    r-=torch.tensor([100*dt])
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

            c+=1
        
        #Logging to wandb after each episode
        wandb.log(
                dict(
                    loss=model.loss,
                    survival_time = t
                ),
                step=ep,
            )

    wandb.finish()

    return model

# Collision handling
def begin_ball_col(arbiter, space, data):
    data["col"]=1
    data["tot_col"]+=1
    return True

def begin_wall_col(arbiter, space, data):
    data["col"]=1
    return True
    
# State is defined by the relative positions and velocities of N nearby balls
def get_state(bodies):
    # Get positions and velocities
    s = [(*tuple(b.position),*tuple(b.velocity)) for b in bodies]

    # unpack into one vector
    s = list(itertools.chain(*s))

    return torch.tensor(s, dtype=torch.float32).to(NN.device)


# Function to flip the velocity when it hits the boundary
def flip_velocity_if_boundary(width,height,object, col):
    # Double if statements ensure - or + velocity when corresponding boundary is hit
    if object.position.x > width - object.radius - 0.5 and col:
        object.velocity = -1*abs(object.velocity.x), object.velocity.y
    elif object.position.x < object.radius + 0.5 and col:
        object.velocity = abs(object.velocity.x), object.velocity.y

    if object.position.y > height - object.radius - 0.5 and col:
        object.velocity = object.velocity.x, -1*abs(object.velocity.y)
    elif object.position.y < object.radius + 0.5 and col:
        object.velocity = object.velocity.x, abs(object.velocity.y)
    col=0

def sim(space, T, dt, model):
    ts = np.arange(0,T,dt)
    bodies = space.bodies
    frame_info = [[tuple(b.position) for b in bodies]]

    c = 0
    sim_per_s = 4

    wall_col = space.add_collision_handler(0,2)
    wall_col.data["col"] = 0
    wall_col.begin = begin_wall_col

    for t in ts:
        
        # get state of N nearest balls
        s = get_state(bodies)
        s = torch.cat((torch.tensor([0,0,width,height]).to("mps"),s))

        # get action from model and apply it for nex, otherwise no velocity change
        with torch.inference_mode():
            if c%(1/dt/sim_per_s) == 0 and not wall_col.data["col"]:
                action_index = model["network"](s).max(0).indices.view(1)
                a_vel, a_ang = model["actions"][action_index]

                # Set velocity values
                x_vel = a_vel*math.cos(a_ang)
                y_vel = a_vel*math.sin(a_ang)

                bodies[-1].velocity = x_vel, y_vel
            
        wall_col.data["col"]=0
        c+=1

        #Step the simulation - KEY
        space.step(dt)

        flip_velocity_if_boundary(width,height,bodies[-1], wall_col.data["col"])

        #log ball positions
        frame_info.append([tuple(b.position) for b in bodies])
    
    return ts[: len(frame_info)], frame_info

if __name__ == "__main__":
    T = 50 # How long to simulate
    dt = 1/200 # we simulate 200 timesteps per second

    run_name = "faithful-sweep-4"

    # Train and save the target network, and set of actions to choose from
    location = "/Users/qarooni2/Documents/Coding/Python/Exercises/Animation/"
    if not run_name:
        simulator = train_model(dt)
        
        network = torch.jit.script(simulator.target_net)
        network.save(location+"models/"+run_name+".pt")

        with open(location+"actions", "wb") as path:
            pickle.dump(simulator.actions, path)
    
    # Load the network and set of actions to use in the simulation
    model = {}
    model["network"] = torch.jit.load(location+"models/"+run_name+".pt")
    with open(location+"actions", "rb") as path:
        model["actions"] = pickle.load(path)
    
    # Initialize and run simulation
    width , height, space = Space.initialize()
    ts, frame_info = sim(
        space, T, dt, model
    )

    bodies = space.bodies

    #Subsamping: render one of 10 below subsamples
    subsampling = 10

    # Prepare figure and axes
    fig, ax = plt.subplots(figsize=(width,height))
    ax.set(xlim=[0,width], ylim=[0,height])
    ax.set_aspect("equal")
    ax.set_position([0,0,1,1])
    fig.set(facecolor="y")

    #Prepare the patches for the balls
    cmap = plt.get_cmap("twilight")
    circles = [plt.Circle((0,0), radius=b.radius, facecolor=cmap(i/len(bodies))) for i,b in enumerate(bodies[:-1])]
    our_guy = plt.Circle((0,0),radius=bodies[-1].radius, facecolor = "Green")
    for c in circles:
        ax.add_patch(c)
        ax.add_patch(our_guy)

    #Draw walls
    for s in space.static_body.shapes:
        ax.plot([s.a.x,s.b.x], [s.a.y,s.b.y], linewidth=2, color="k")

    #animation function. Called for each frame, passing an entry in positions
    def drawframe(p):
        for i, c in enumerate (circles):
            c.set_center(p[i])
        our_guy.set_center(p[-1])
        return circles + [our_guy]
    
    anim = animation.FuncAnimation(
        fig,
        drawframe,
        frames=frame_info[::subsampling],
        interval = dt * subsampling * 1000,
        blit=True
    )

    plt.show()
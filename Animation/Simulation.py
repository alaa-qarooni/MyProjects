import Space
import NN

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

import os
import math
import torch
import itertools
import pickle
from collections import deque

import pymunk
from pymunk.vec2d import Vec2d

PYTORCH_ENABLE_MPS_FALLBACK=1

def train_model(dt, n_episodes=10,episode_length=10):
    
    # state is composed of the positions and velocities of N-1 nearest dynamic balls
    # REALTIVE TO 1 kinematic ball.
    # The kinematic balls will take the action from the target network.
    # Action space is velocity and direction kinematic ball can take: 0 rad - 2pi rad
    N = 21
    action_space = list(itertools.product(np.arange(-3,4,0.5),np.arange(-3,3.1,0.5)))
    state_space = [0]*N*4
    model = NN.simulator(state_space,action_space)

    for ep in range(n_episodes):
        width,height,space = Space.initialize()

        running_r = r = torch.tensor([0.])
        
        balls = space.bodies[:-1]
        our_guy = space.bodies[-1]

        s = get_state(N,space.bodies)

        ch = space.add_wildcard_collision_handler(0)
        ch.data["col"] = 0
        ch.data["tot_col"] = 0
        ch.begin = begin
        col = 0

        for t in np.arange(0,episode_length,dt):
            ch.data["col"] = 0
            # select action
            a = model.select_action(s, decay=1/dt * episode_length * n_episodes / 2)

            # Apply action to velocity
            a_vel, a_ang = model.actions[a]
            x_vel = a_vel*math.cos(a_ang)
            y_vel = a_vel*math.sin(a_ang)

            our_guy.velocity = x_vel,y_vel

            # Flip velocity if it hits the boundary
            flip_velocity_if_boundary(width,height,our_guy)
            
            # Step in simulation
            space.step(dt)
            
            # Penalize based on distances of N nearest balls
            r += torch.tensor(sum([-50*dt*math.exp(-5*Vec2d.get_distance(our_guy.position,(s[x],s[y]))) for x,y in zip(range(0,39,4),range(1,39,4))]))
            
            # Heavy penalty introduced for collisions with either ball or wall.
            if ch.data["col"]:
                r+=torch.tensor([-100*dt])

            # Time-based reward for staying in the space
            r += torch.tensor([dt])

            # Get next state
            s_prime = get_state(N,space.bodies)
            
            running_r += r

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

        # Survival time in seconds
        print(f'Kin ball {ep} collided with dyn balls or wall {ch.data["tot_col"]} times')

    return model

# Collision handling
def begin(arbiter, space, data):
    data["col"]=1
    data["tot_col"]+=1
    return True
    
# State is defined by the relative positions and velocities of N nearby balls
def get_state(N, bodies):
    #Get nearest N balls
    balls = sorted(bodies,key=lambda b: Vec2d.get_distance(bodies[-1].position,b.position))[:N]
    
    # Make kinetic ball features the last to follow convention (sorted func above made it first)
    balls[0], balls[-1] = balls[-1], balls[0]

    # Get positions and velocities
    s = [(*tuple(b.position),*tuple(b.velocity)) for b in balls]

    # unpack into one vector
    s = list(itertools.chain(*s))

    return torch.tensor(s ,dtype=torch.float32).to(NN.device)


# Function to flip the velocity when it hits the boundary
def flip_velocity_if_boundary(width,height,object):
    # Double if statements ensure - or + velocity when corresponding boundary is hit
    if object.position.x > width - object.radius - 0.1:
        object.velocity = -1*abs(object.velocity.x), object.velocity.y
    elif object.position.x < object.radius + 0.1:
        object.velocity = abs(object.velocity.x), object.velocity.y

    if object.position.y > height - object.radius - 0.1:
        object.velocity = object.velocity.x, -1*abs(object.velocity.y)
    elif object.position.y < object.radius + 0.1:
        object.velocity = object.velocity.x, abs(object.velocity.y)

def sim(space, T, dt, model):
    ts = np.arange(0,T,dt)
    bodies = space.bodies
    frame_info = [[tuple(b.position) for b in bodies]]

    prev_vel = tuple(bodies[-1].velocity)
    c = 0
    for t in ts:
        
        # get state of N nearest balls
        s = get_state(21, bodies)

        # If nearby balls, get action from model and apply it for the next 50 * dt seconds (0.25 seconds), otherwise no velocity change
        if any([Vec2d.get_distance(bodies[-1].position,b.position)<bodies[-1].radius+b.radius+5 for b in bodies[:-1]]):
            if c == 0:
                action_index = model["network"](s).max(0).indices.view(1)
                a_v,a_ang = model["actions"][action_index]

                # Set velocity values
                x_vel = a_v*math.cos(a_ang)
                y_vel = a_v*math.sin(a_ang)

                bodies[-1].velocity = x_vel, y_vel
            c+=1
            if c == 50:
                c=0
        else:
            c=0
        
        flip_velocity_if_boundary(width,height,bodies[-1])

        #Step the simulation - KEY
        space.step(dt)

        #log ball positions
        frame_info.append([tuple(b.position) for b in bodies])
    
    return ts[: len(frame_info)], frame_info

if __name__ == "__main__":
    T = 50 # How long to simulate
    dt = 1/200 # we simulate 200 timesteps per second
    
    # Train and save the target network, and set of actions to choose from
    if not os.path.isfile("Exercises/Animation/model.pt"):
        simulator = train_model(dt)
        
        network = torch.jit.script(simulator.target_net)
        network.save("Exercises/Animation/model.pt")

        with open("Exercises/Animation/actions", "wb") as path:
            pickle.dump(simulator.actions, path)
    
    # Load the network and set of actions to use in the simulation
    model = {}
    model["network"] = torch.jit.load("Exercises/Animation/model.pt")
    with open("Exercises/Animation/actions", "rb") as path:
        model["actions"] = pickle.load(path)
    
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

import Space
import NN

import numpy as np
import torch
from pynput import keyboard
from tqdm import tqdm

import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import pygame

PYTORCH_ENABLE_MPS_FALLBACK=1
simulate_flag = False

def on_press(key):
    global simulate_flag
    try:
        if key.char == 's':
            simulate_flag = True
    except AttributeError:
        pass

# Start listener in background
listener = keyboard.Listener(on_press=on_press)
listener.start()

def train_model(width, height, dt, n_episodes=1800, episode_length=20, pretrained_weights=None):
    # No need for discrete action space - PPO handles continuous actions
    state_space = [0]*16
    avg_reward = survival_time = 0
    progress_bar = None
    
    # Create PPO agent with 2 continuous actions (x_vel, y_vel)
    model = NN.PPOAgent(len(state_space), 2,  # 2 continuous actions
                       gamma=0.99, lr=3e-4, batch_size=64, 
                       n_layers=3, dim=256, dropout_rate=0.1)
    
    difficulty_schedule = {
        0:    {"n_balls": 5,  "ball_size": 30, "ball_speed": 0},
        500:  {"n_balls": 6,  "ball_size": 28, "ball_speed": 0},    # Slower increases
        1000: {"n_balls": 7,  "ball_size": 26, "ball_speed": 20},   # Tiny speed increases
        1500: {"n_balls": 8,  "ball_size": 24, "ball_speed": 40},
        2000: {"n_balls": 9,  "ball_size": 22, "ball_speed": 60},
        2500: {"n_balls": 10, "ball_size": 20, "ball_speed": 80},   # Much slower progression
        3000: {"n_balls": 12, "ball_size": 20, "ball_speed": 100},
        3500: {"n_balls": 15, "ball_size": 20, "ball_speed": 150},  # Final difficulty
    }
    
    for ep in range(n_episodes):
        current_difficulty = max([k for k in difficulty_schedule.keys() if k <= ep])
        params = difficulty_schedule[current_difficulty]
        
        width, height, space = Space.initialize(width, height, 
                                              n_balls=params["n_balls"], 
                                              ball_size=params["ball_size"],
                                              ball_speed=params["ball_speed"])
        our_guy = list(space.bodies)[-1]
        s, _ = get_state(space)
        
        steps = 0

        global simulate_flag
        if simulate_flag:
            model.actor.eval()
            simulate(width,height,space,model.actor,time=20)
            model.actor.train()
            simulate_flag = False

        if ep % 100 == 0:
            if progress_bar is not None:
                progress_bar.close()
            print(f"Eps {ep-100}-{ep}: "
                f"Avg Reward={avg_reward/100:.3f}, "
                f"Avg Survival={np.mean(survival_time/100):.1f}s")
            progress_bar = tqdm(total=int(episode_length/dt))
            avg_reward=survival_time=0

        if progress_bar is not None:
            progress_bar.reset()
            progress_bar.set_description(f"Episode {ep+1}")
        
        for t in np.arange(0, episode_length, dt):
            
            # Select continuous action using PPO
            if steps % steps_per_update ==0:
                action, prob, val = model.choose_action(s.cpu().numpy())
            
            steps += 1
            
            # Apply continuous action directly (scaled from [-1,1] to velocity range)
            x_vel, y_vel = action * 200  # Scale to [-200, 200]
            our_guy.velocity = x_vel, y_vel
            
            # Step simulation
            space.step(dt)
            
            # Get next state and reward
            s_prime, distance_reward = get_state(space)
            
            # Calculate reward
            if not distance_reward:  # Collision
                reward = -10.0
                done = True
                s_prime = None
                survival_time+=t
            else:
                if t >= episode_length - dt:  # Episode completed
                    reward = 0.05 + distance_reward + 10.0  # Completion bonus
                    done = True
                    survival_time+=t
                else:
                    reward = 0.05 + distance_reward
                    done = False
            
            # Store transition
            model.store_transition(s.cpu().numpy(), action, prob, val, reward, done)
            
            # Learn from experiences
            if steps % steps_per_update == 0:
                model.learn()
            
            # Move to next state
            s = s_prime
            avg_reward += reward
            
            if progress_bar:
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'avg_reward': avg_reward/(ep%100 + 1),
                    'avg_survival_time': survival_time/(ep%100 + 1)
                })

            if done:
                break
    

    progress_bar.close()
    return model
    
# State is defined by the normalized position and velocity of the ball in space and its ray vision
def get_state(space):
    our_guy = list(space.bodies)[-1]

    pos = [our_guy.position.x/width, our_guy.position.y/height]
    vel = [our_guy.velocity.x/500, our_guy.velocity.y/500]

    rays, _ = get_ray_trace(space)
    reward = 0.025*sum(rays)/len(rays) if min(rays)>0 else 0

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

def simulate(width, height, space, actor_model, time=float("inf")):
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    draw_options = pymunk.pygame_util.DrawOptions(screen)
    actor_model.eval()

    running = True
    count = 0
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if count * dt > time:
            running = False
        
        # Get continuous action from trained actor
        state_tensor = get_state(space)[0].to(NN.device)
        if count%steps_per_update==0:
            action = NN.select_action_trained(actor_model, state_tensor)
        
        our_guy = list(space.bodies)[-1]
        count += 1
        
        # Apply continuous velocity
        x_vel, y_vel = action * 200
        our_guy.velocity = x_vel, y_vel
        
        # Edge control and simulation step
        edge_control(width, height, our_guy)
        space.step(dt)
        
        # Rendering code remains the same...
        screen.fill((255, 255, 255))
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
        
        pygame.display.flip()
        clock.tick(1/dt)

    pygame.quit()

if __name__ == "__main__":
    dt = 1/60
    steps_per_update = 10

    # Initialize simulation
    width, height, space = Space.initialize(1280, 720, n_balls=20, ball_size=20, ball_speed=0)

    # Train PPO model
    run_name = "first"
    location = "animation_ppo/models/"
    weights_path = location + run_name + ".pt"
    train = True
    if train:
        agent = train_model(width, height, dt, pretrained_weights=None)
        
        # Save only the actor network for inference
        torch.save(agent.actor.state_dict(), weights_path)
    
    # Load for inference
    state_dim = 16  # Your state dimension
    actor = NN.ActorNetwork(state_dim, 2, 3, 256, 0.1).to(NN.device)  # 2 continuous actions
    actor.load_state_dict(torch.load(weights_path))
    
    simulate(width, height, space, actor)

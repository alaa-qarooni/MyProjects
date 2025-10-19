import warnings
warnings.filterwarnings("ignore")
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

listener = keyboard.Listener(on_press=on_press)
listener.start()

def train_model(width, height, dt, n_episodes=8000, episode_length=60, pretrained_weights=None):
    state_space = [0]*28
    avg_reward = survival_time = 0
    progress_bar = None
    level = 1
    target_ep = 0
    
    model = NN.PPOAgent(len(state_space), 2,
                        pretrained_weights,
                        gamma=0.995,      # Slightly increased
                        lr=1e-4,          # Reduced learning rate for stability
                        batch_size=128,    # Increased batch size
                        ppo_epochs=6,     # More PPO epochs
                        clip=0.15,        # Tighter clipping
                        gae_lambda=0.92,  # Adjusted GAE
                        dim=512)          # Match new hidden size
    
    
    difficulty_schedule = [
        # Phase 1: Learn to avoid static obstacles (slightly harder)
        {"n_balls": 3, "ball_size": 40, "ball_speed": 0},
        {"n_balls": 6, "ball_size": 38, "ball_speed": 0},
        {"n_balls": 8, "ball_size": 36, "ball_speed": 10},

        {"n_balls": 10, "ball_size": 34, "ball_speed": 20},
        {"n_balls": 12, "ball_size": 32, "ball_speed": 40},
        {"n_balls": 14, "ball_size": 30, "ball_speed": 60},

        {"n_balls": 16, "ball_size": 28, "ball_speed": 80},
        {"n_balls": 18, "ball_size": 26, "ball_speed": 100}

    ]
    
    for ep in range(n_episodes):
        params = difficulty_schedule[min(level-1, len(difficulty_schedule)-1)]
        
        width, height, space, target_body, target_shape, our_guy_body, our_guy_shape = Space.initialize(
            width, height, 
            n_balls=params["n_balls"], 
            ball_size=params["ball_size"],
            ball_speed=params["ball_speed"]
        )
        
        s, _ = get_state(space, target_body)
        
        global simulate_flag
        if simulate_flag:
            model.model.eval()
            simulate(width, height, space, target_body, target_shape, our_guy_body, our_guy_shape, model.model, time=20)
            model.model.train()
            simulate_flag = False

        if ep % 50 == 0:
            if progress_bar is not None:
                progress_bar.close()

            if ep > 0:
                print(f"Eps {ep-50}-{ep}: "
                    f"Avg Reward={avg_reward/50:.3f}, "
                    f"Avg Survival={survival_time/50:.1f}s, "
                    f"Targets Reached={target_ep/50:.1f}")
            
            if round(target_ep / 50,1) >= 3 and level < len(difficulty_schedule):
                print(f"Advancing to level {level + 1}!")
                level += 1

            target_ep=0
            
            progress_bar = tqdm(total=int(episode_length/dt))
            avg_reward = survival_time = target_ep = 0

        steps = 0
        episode_reward = 0
        target_reached = 0
        last_distance = Vec2d.get_distance(our_guy_body.position, target_body.position)
        bump_count = 0

        progress_bar.reset()
        progress_bar.set_description(f"Episode {ep+1} (Lvl {level})")
        
        for t in np.arange(0, episode_length, dt):
            if steps % steps_per_update == 0:
                action, prob, val = model.choose_action(s.cpu().numpy())
            
            steps += 1
            
            x_vel, y_vel = action * 200
            our_guy_body.velocity = x_vel, y_vel
            
            space.step(dt)
            
            s_prime, rays = get_state(space, target_body)
            current_distance = Vec2d.get_distance(our_guy_body.position, target_body.position)
            
            # Replace the entire reward calculation section with:
            reward = 0
            done = False

            # Target reached reward (keep strong but add progress bonus)
            if current_distance < (our_guy_shape.radius + target_shape.radius):
                reward += 100.0
                target_reached += 1
                target_ep += 1
                target_body.position = (np.random.randint(100, width-100), np.random.randint(100, height-100))
                last_distance = current_distance

            # Progressive distance reward - more reward for getting closer
            distance_reward = (last_distance - current_distance) * 0.1
            # Add non-linear bonus for being close to target
            if current_distance < 200:  # Close to target
                distance_reward *= 2.0
            reward += distance_reward

            # Time penalty (reduced to encourage exploration)
            reward -= 0.05

            # Strategic obstacle avoidance reward
            min_ray = min(rays)
            safe_distance = 0.1  # Increased safe distance

            if min_ray > safe_distance:
                # Reward for maintaining safe distance from obstacles
                reward += 0.2
            elif min_ray < 0.05:  # Collision
                reward -= 15.0  # Reduced penalty to encourage risk-taking
                bump_count += 1
                if bump_count > 3:
                    done = True
            else:
                # Gradual penalty as getting closer to obstacles
                proximity_penalty = -2.0 * (safe_distance - min_ray) / safe_distance
                reward += proximity_penalty
                

            # Progress bonus when navigating through tight spaces
            if min_ray < 0.1 and distance_reward > 0:
                reward += 5.0  # Bonus for making progress while near obstacles

            # Episode completion bonus
            if t >= episode_length - dt:
                reward += 8.0 * target_reached  # Increased completion bonus
                done = True

            last_distance = current_distance
            
            model.store_transition(s.cpu().numpy(), action, prob, val, reward, done)
            episode_reward += reward
            
            if done:
                s = get_state(space, target_body)[0]
                survival_time += t
                break
            else:
                s = s_prime
            
            if progress_bar:
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'reward': episode_reward,
                    'time': t,
                    'targets': target_reached,
                    'average': f"{target_ep / (ep%50) if (ep%50)!=0 else 0:.2f}"
                })

        if len(model.episode_states) >= model.mini_batch_size:
            model.learn()
        
        avg_reward += episode_reward

    if progress_bar:
        progress_bar.close()
    return model

def get_state(space, target):
    our_guy = list(space.bodies)[-1]
    
    vel = [our_guy.velocity.x/500, our_guy.velocity.y/500]
    rays, _ = get_ray_trace(space, target)
    
    target_dx = (target.position.x - our_guy.position.x) / width
    target_dy = (target.position.y - our_guy.position.y) / height
    
    # Keep original 28-dimensional state for compatibility
    s = vel + rays + [target_dx, target_dy]
    
    return torch.tensor(s, dtype=torch.float32).to(NN.device), rays 

def get_ray_trace(space, target_body, sensors=24, sensor_range=150):
    our_guy = list(space.bodies)[-1]
    vals = [1.0]*sensors
    contact_points = []
    for i in range(sensors):
        start_pos = our_guy.position + (Space.get_body_radius(our_guy)+2)*Vec2d(np.cos(i*2*np.pi/sensors),np.sin(i*2*np.pi/sensors))
        end_pos = start_pos + sensor_range*Vec2d(np.cos(i*2*np.pi/sensors),np.sin(i*2*np.pi/sensors))
        
        # Remove the filter line and just use default filter
        hit = space.segment_query_first(start_pos, end_pos, 0, pymunk.ShapeFilter())
        if hit and hit.shape.body != target_body:
            vals[i] = round(hit.alpha,6)
            if vals[i]==0:
                contact_points.append(start_pos)
            else:
                contact_points.append(hit.point)
    
    return vals, contact_points

def edge_control(width,height,our_guy):
    radius = Space.get_body_radius(our_guy)
    if round(our_guy.position.x - radius) < 15:
        our_guy.velocity = Vec2d(0,our_guy.velocity.y)
    if round(our_guy.position.y - radius) < 15:
        our_guy.velocity = Vec2d(our_guy.velocity.x,0)
    if round(our_guy.position.x + radius) > width-15:
        our_guy.velocity = Vec2d(0,our_guy.velocity.y)
    if round(our_guy.position.y + radius) > height-15:
        our_guy.velocity = Vec2d(our_guy.velocity.x,0)

def simulate(width, height, space, target_body, target_shape, our_guy_body, our_guy_shape, model, time=float("inf")):
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    model.eval()

    running = True
    paused = False
    count = 0
    targets_reached = 0
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused  # Toggle pause
        
        # Skip simulation step if paused
        if not paused:
            state_tensor = get_state(space, target_body)[0].to(NN.device)
            if count % steps_per_update == 0:
                action = NN.select_action_trained(model, state_tensor.unsqueeze(0))
            
            count += 1
            
            x_vel, y_vel = action * 200
            our_guy_body.velocity = x_vel, y_vel
            
            # Check for target collision
            our_guy_radius = Space.get_body_radius(our_guy_body)
            target_radius = Space.get_body_radius(target_body)
            current_distance = Vec2d.get_distance(our_guy_body.position, target_body.position)
            if current_distance < (our_guy_radius + target_radius):
                targets_reached += 1
                target_body.position = (np.random.randint(100, width-100), np.random.randint(100, height-100))
            
            edge_control(width, height, our_guy_body)
            space.step(dt)
        
            # Manual rendering without debug draw
            screen.fill((255, 255, 255))
            
            # Draw walls
            for shape in space.shapes:
                if isinstance(shape, pymunk.Segment):
                    body = shape.body
                    p1 = pymunk.pygame_util.to_pygame(shape.a + body.position, screen)
                    p2 = pymunk.pygame_util.to_pygame(shape.b + body.position, screen)
                    pygame.draw.line(screen, (0, 0, 0), p1, p2, 5)
            
            # Draw balls with custom colors
            for shape in space.shapes:
                if isinstance(shape, pymunk.Circle):
                    body = shape.body
                    pos = pymunk.pygame_util.to_pygame(body.position, screen)
                    radius = int(shape.radius)
                    
                    # Determine color
                    if hasattr(shape, 'color'):
                        # Convert from RGBA (0-1.0) to RGB (0-255)
                        r, g, b, a = shape.color
                        color = (int(r * 255), int(g * 255), int(b * 255))
                    else:
                        # Default colors based on body type
                        if body.body_type == pymunk.Body.STATIC:
                            color = (0, 255, 0)  # Green for target
                        elif body == our_guy_body:
                            color = (0, 0, 255)  # Blue for our guy
                        else:
                            color = (255, 0, 0)  # Red for obstacles
                    
                    pygame.draw.circle(screen, color, pos, radius)
                    pygame.draw.circle(screen, (0, 0, 0), pos, radius, 2)
            
            vals, contact_points = get_ray_trace(space, target_body)  # Add target_body parameter
            
            for contact in contact_points:
                contact_pos = pymunk.pygame_util.to_pygame(contact, screen)
                pygame.draw.line(screen, (255,0,0), our_guy_body.position, contact_pos, 1)

            rays = pygame.font.SysFont("Consolas", 16).render(f"Velocity: {'%.0f, %.0f' % (our_guy_body.velocity.x, our_guy_body.velocity.y)}", True, (0, 0, 0))
            screen.blit(rays, (10, 10))
            position = pygame.font.SysFont("Consolas", 16).render(f"Sensors: {[f'{v:.2f}' for v in vals[:12]]}", True, (0, 0, 0))
            screen.blit(position, (10,30))
            position = pygame.font.SysFont("Consolas", 16).render(f"         {[f'{v:.2f}' for v in vals[12:]]}", True, (0, 0, 0))
            screen.blit(position, (10,50))
            position = pygame.font.SysFont("Consolas", 16).render(f"{count*dt:.1f}s", True, (0, 0, 0))
            screen.blit(position, (width-100,height-50))
            targets_text = pygame.font.SysFont("Consolas", 16).render(f"Targets: {targets_reached}", True, (0, 0, 0))
            screen.blit(targets_text, (10, 70))
            
            distance = Vec2d.get_distance(our_guy_body.position, target_body.position)
            distance_text = pygame.font.SysFont("Consolas", 16).render(f"Distance: {distance:.1f}", True, (0, 0, 0))
            screen.blit(distance_text, (10, 90))
            
            pygame.display.flip()
            clock.tick(1/dt)

    pygame.quit()

if __name__ == "__main__":
    dt = 1/60
    steps_per_update = 10

    width, height, space, target_body, target_shape, our_guy_body, our_guy_shape = Space.initialize(
        1280, 720, n_balls=10, ball_size=20, ball_speed=50
    )

    run_name = "target_seeker_third"
    location = "animation_ppo/models/"
    weights_path = [location + run_name + "_actor.pt", location + run_name + "_critic.pt"]
    with_weights = False
    train = True
    if train:
        agent = train_model(width, height, dt, pretrained_weights=weights_path if with_weights else [None,None])
        torch.save(agent.model.state_dict(), weights_path[0])
    
    state_dim = 28
    model = NN.ActorCritic(state_dim, 2, hidden_size=512).to(NN.device)
    model.load_state_dict(torch.load(weights_path[0]))
    
    simulate(width, height, space, target_body, target_shape, our_guy_body, our_guy_shape, model, time=50)
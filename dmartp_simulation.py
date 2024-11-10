import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

class Agent:
    def __init__(self, agent_id, start_pos, goal_pos, max_speed=1.0, radius=0.5):
        self.agent_id = agent_id
        self.start_pos = np.array(start_pos, dtype=float)  # Store start position
        self.position = np.array(start_pos, dtype=float)
        self.goal = np.array(goal_pos, dtype=float)
        self.velocity = np.zeros(2)
        self.max_speed = max_speed
        self.radius = radius
        self.path = [self.position.copy()]  # For visualization

    def sense(self, agents, obstacles, perception_range):
        # Sense nearby agents and obstacles
        sensed_agents = []
        for agent in agents:
            if agent.agent_id != self.agent_id:
                dist = np.linalg.norm(self.position - agent.position)
                if dist <= perception_range:
                    sensed_agents.append(agent)
        sensed_obstacles = []
        for obs in obstacles:
            dist = np.linalg.norm(self.position - obs)
            if dist <= perception_range:
                sensed_obstacles.append(obs)
        return sensed_agents, sensed_obstacles

    def predict(self, sensed_agents):
        # Simple prediction: assume other agents maintain their current velocity
        predictions = []
        for agent in sensed_agents:
            predicted_position = agent.position + agent.velocity
            predictions.append((agent, predicted_position))
        return predictions

    def plan_path(self, predictions, sensed_obstacles):
        # Simple path planning towards the goal while avoiding predicted positions
        direction_to_goal = self.goal - self.position
        dist_to_goal = np.linalg.norm(direction_to_goal)
        
        if dist_to_goal > 0:
            direction_to_goal /= dist_to_goal
            
        # Avoid predicted positions
        avoidance_vector = np.zeros(2)
        for agent, predicted_pos in predictions:
            diff = self.position - predicted_pos
            dist = np.linalg.norm(diff)
            if dist < self.radius * 2 and dist > 0:  # Add check for zero distance
                avoidance_vector += diff / dist

        # Avoid obstacles
        for obs in sensed_obstacles:
            diff = self.position - obs
            dist = np.linalg.norm(diff)
            if dist < self.radius * 2 and dist > 0:  # Add check for zero distance
                avoidance_vector += diff / dist

        # Combine directions
        new_velocity = direction_to_goal + avoidance_vector
        speed = np.linalg.norm(new_velocity)
        
        if speed > 0:
            new_velocity = (new_velocity / speed) * min(self.max_speed, speed)
            
        self.velocity = new_velocity

    def update_position(self, dt=1.0):
        self.position += self.velocity * dt
        self.path.append(self.position.copy())

class Environment:
    def __init__(self, agents, obstacles):
        self.agents = agents
        self.obstacles = obstacles  # Obstacles are represented as points for simplicity

    def update(self):
        for agent in self.agents:
            sensed_agents, sensed_obstacles = agent.sense(self.agents, self.obstacles, perception_range=5.0)
            predictions = agent.predict(sensed_agents)
            agent.plan_path(predictions, sensed_obstacles)
        for agent in self.agents:
            agent.update_position()

    def is_simulation_complete(self):
        for agent in self.agents:
            if np.linalg.norm(agent.position - agent.goal) > 0.5:
                return False
        return True

    def simulate(self, max_steps=100):
        step = 0
        while not self.is_simulation_complete() and step < max_steps:
            self.update()
            step += 1
        return step

    def visualize(self):
        plt.figure(figsize=(10, 10))
        
        # Plot obstacles
        if self.obstacles:
            obstacles = np.array(self.obstacles)
            plt.scatter(obstacles[:, 0], obstacles[:, 1], c='black', marker='s', label='Obstacles')
            
        # Plot agent paths
        colors = ['b', 'g', 'r', 'c', 'm', 'y']  # Different colors for different agents
        for i, agent in enumerate(self.agents):
            path = np.array(agent.path)
            color = colors[i % len(colors)]
            
            # Plot path
            plt.plot(path[:, 0], path[:, 1], color=color, label=f'Agent {agent.agent_id}')
            
            # Plot start position
            plt.scatter(agent.start_pos[0], agent.start_pos[1], color=color, marker='o')
            
            # Plot goal position
            plt.scatter(agent.goal[0], agent.goal[1], color=color, marker='x')
            
            # Add arrow to show final direction
            if len(path) > 1:
                plt.arrow(path[-2, 0], path[-2, 1], 
                         path[-1, 0] - path[-2, 0], path[-1, 1] - path[-2, 1],
                         head_width=0.3, head_length=0.5, fc=color, ec=color)

        plt.legend()
        plt.grid(True)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Multi-Agent Navigation Paths')
        
        # Set equal aspect ratio
        plt.axis('equal')
        plt.show()

# Example Usage
def main():
    # Define agents with different starting positions and goals
    agents = [
        Agent(agent_id=1, start_pos=(0, 0), goal_pos=(10, 10)),
        Agent(agent_id=2, start_pos=(10, 0), goal_pos=(0, 10)),
        Agent(agent_id=3, start_pos=(5, 0), goal_pos=(5, 10))
    ]
    
    # Define obstacles as a list of points
    obstacles = [
        (5, 5), (6, 5), (5, 6)
    ]
    
    # Create and run simulation
    env = Environment(agents, obstacles)
    steps_taken = env.simulate(max_steps=50)
    print(f"Simulation completed in {steps_taken} steps")
    
    # Visualize results
    env.visualize()

if __name__ == "__main__":
    main()
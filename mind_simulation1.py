import networkx as nx
import matplotlib.pyplot as plt
import random
import time
import numpy as np

# --- Configuration ---
MAX_AGE = 99
INITIAL_AGE = 0.1
TIME_STEP = 0.1  # How much age advances each frame
ANIMATION_PAUSE = 0.1 # Pause between frames

NODE_ID_COUNTER = 0

# Prominence
INITIAL_PROMINENCE = 0.5
REINFORCE_AMOUNT = 0.2
PROMINENCE_DECAY_RATE = 0.01 # Per time step if not reinforced
PROMINENCE_PRUNE_THRESHOLD = 0.05
MAX_PROMINENCE = 1.0

# Event Probabilities (per time step)
NATURE_EVENT_PROB = 0.3
SUPERSTRUCTURE_EVENT_PROB = 0.25
INTERNAL_CONNECTION_PROB = 0.2

# Colors
COLOR_MAP = {
    "nature": "green",
    "superstructure": "blue",
    "internal": "purple",
    "mixed": "gray" # For nodes with significant mixed influences
}
DEFAULT_NODE_COLOR = "lightgray"
DEFAULT_EDGE_COLOR = "gray"

# Potential thoughts (simplified)
NATURE_THOUGHTS = ["Warmth", "Cold", "Hunger", "Thirst", "Touch", "Sound", "Light", "Mother's Face", "Pain", "Pleasure"]
SUPERSTRUCTURE_THOUGHTS = ["Language_Unit", "Rule", "Story", "Belief", "Skill", "Tool_Concept", "Social_Norm", "Abstract_Idea"]

# --- Helper Functions ---
def get_new_node_id():
    global NODE_ID_COUNTER
    NODE_ID_COUNTER += 1
    return NODE_ID_COUNTER

def get_dominant_influence_color(influences_dict):
    if not influences_dict:
        return DEFAULT_NODE_COLOR
    
    # Normalize influences if they don't sum to 1 (they should ideally)
    total_influence = sum(influences_dict.values())
    if total_influence == 0: return DEFAULT_NODE_COLOR
    
    normalized_influences = {k: v / total_influence for k, v in influences_dict.items()}
    
    # Check for a clear dominant influence
    dominant_type = max(normalized_influences, key=normalized_influences.get)
    if normalized_influences[dominant_type] > 0.6: # Threshold for being "dominant"
        return COLOR_MAP.get(dominant_type, DEFAULT_NODE_COLOR)
    
    # If mixed, count distinct influences
    present_influences = [inf for inf in ["nature", "superstructure", "internal"] if influences_dict.get(inf, 0) > 0.1]
    if len(present_influences) > 1:
        return COLOR_MAP["mixed"]
    elif present_influences:
        return COLOR_MAP.get(present_influences[0], DEFAULT_NODE_COLOR)
        
    return DEFAULT_NODE_COLOR

def generate_thought_label(base_label, existing_labels):
    count = 1
    label = base_label
    while label in existing_labels:
        label = f"{base_label}_{count}"
        count += 1
    return label

# --- Simulation Class ---
class MindSimulation:
    def __init__(self):
        self.graph = nx.DiGraph() # Using DiGraph for directed thoughts/associations
        self.current_age = 0.0
        self.node_labels = set() # To ensure unique labels if needed

    def initialize_mind(self):
        self.current_age = INITIAL_AGE
        # Initial thoughts (small, not blank slate)
        
        # "Self" - somewhat internal, but also based on bodily sensation (nature)
        self_id = self.add_thought("Self", {"internal": 0.5, "nature": 0.5}, prominence=0.7)
        
        # Basic sensory input
        sensation_id = self.add_thought("Sensation_Comfort", {"nature": 1.0}, prominence=0.6)
        discomfort_id = self.add_thought("Sensation_Discomfort", {"nature": 1.0}, prominence=0.6)

        # Connect them
        self.add_connection(sensation_id, self_id, "nature")
        self.add_connection(discomfort_id, self_id, "nature")
        
        print(f"Mind initialized at age {self.current_age:.1f} with {self.graph.number_of_nodes()} thoughts.")

    def add_thought(self, base_label, influences, prominence=INITIAL_PROMINENCE, existing_node_to_connect=None, connection_influence_type=None):
        node_id = get_new_node_id()
        label = generate_thought_label(base_label, self.node_labels)
        self.node_labels.add(label)

        self.graph.add_node(
            node_id,
            label=label,
            influences=influences.copy(), # Store proportions
            prominence=min(prominence, MAX_PROMINENCE),
            age_created=self.current_age,
            last_reinforced=self.current_age,
            size=prominence * 500 + 100 # For visualization
        )
        # print(f"Age {self.current_age:.1f}: Added thought '{label}' ({node_id}) influenced by {influences}")

        if existing_node_to_connect and self.graph.has_node(existing_node_to_connect) and connection_influence_type:
            self.add_connection(existing_node_to_connect, node_id, connection_influence_type)
            # Also make a connection back, perhaps weaker, or let it form naturally later
            # self.add_connection(node_id, existing_node_to_connect, connection_influence_type, weight=0.5)
        return node_id

    def add_connection(self, source_id, target_id, influence_type, weight=1.0):
        if self.graph.has_node(source_id) and self.graph.has_node(target_id) and source_id != target_id:
            if not self.graph.has_edge(source_id, target_id):
                self.graph.add_edge(source_id, target_id, influence_type=influence_type, color=COLOR_MAP.get(influence_type, DEFAULT_EDGE_COLOR), weight=weight)
                # print(f"Age {self.current_age:.1f}: Connected {self.graph.nodes[source_id]['label']} -> {self.graph.nodes[target_id]['label']} via {influence_type}")


    def reinforce_thought(self, node_id, reinforcing_influence_type=None, amount=REINFORCE_AMOUNT):
        if self.graph.has_node(node_id):
            node = self.graph.nodes[node_id]
            node['prominence'] = min(MAX_PROMINENCE, node['prominence'] + amount)
            node['last_reinforced'] = self.current_age
            node['size'] = node['prominence'] * 500 + 100

            # Update influences if a specific type is reinforcing
            if reinforcing_influence_type:
                current_total_influence = sum(node['influences'].values())
                # Add to the specific influence, then re-normalize if needed (or just let one grow)
                node['influences'][reinforcing_influence_type] = node['influences'].get(reinforcing_influence_type, 0) + amount * 0.5 # Influence update is smaller
                
                # Simple re-normalization (optional, can lead to more 'mixed' nodes)
                new_total_influence = sum(node['influences'].values())
                if new_total_influence > 0 :
                    for inf_type in node['influences']:
                         node['influences'][inf_type] /= new_total_influence
                
            # print(f"Age {self.current_age:.1f}: Reinforced thought '{node['label']}' ({node_id}). New prominence: {node['prominence']:.2f}")


    def apply_events(self):
        # --- Nature Event ---
        if random.random() < NATURE_EVENT_PROB and self.graph.number_of_nodes() > 0:
            new_thought_label = random.choice(NATURE_THOUGHTS)
            influences = {"nature": 1.0}
            
            # Try to connect to an existing prominent node, or a nature-influenced one
            target_node = self.get_connection_target(preferred_influence="nature")
            
            new_node_id = self.add_thought(new_thought_label, influences, 
                                           existing_node_to_connect=target_node, 
                                           connection_influence_type="nature")
            if target_node: # Also reinforce the target
                self.reinforce_thought(target_node, "nature", REINFORCE_AMOUNT / 2)


        # --- Superstructure Event ---
        if random.random() < SUPERSTRUCTURE_EVENT_PROB and self.graph.number_of_nodes() > 0:
            new_thought_label = random.choice(SUPERSTRUCTURE_THOUGHTS)
            influences = {"superstructure": 1.0}
            
            target_node = self.get_connection_target(preferred_influence="superstructure")

            new_node_id = self.add_thought(new_thought_label, influences, 
                                           existing_node_to_connect=target_node, 
                                           connection_influence_type="superstructure")
            if target_node: # Also reinforce the target
                self.reinforce_thought(target_node, "superstructure", REINFORCE_AMOUNT / 2)

        # --- Internal Connection ---
        if random.random() < INTERNAL_CONNECTION_PROB and self.graph.number_of_nodes() >= 2:
            nodes_list = list(self.graph.nodes())
            node1_id, node2_id = random.sample(nodes_list, 2)
            
            # Connect if not already connected and if they are somewhat prominent
            if not self.graph.has_edge(node1_id, node2_id) and \
               self.graph.nodes[node1_id]['prominence'] > 0.3 and \
               self.graph.nodes[node2_id]['prominence'] > 0.3:
                self.add_connection(node1_id, node2_id, "internal")
                # Maybe create a new "bridge" thought? For simplicity, direct connection for now.
                # Reinforce both slightly due to association
                self.reinforce_thought(node1_id, "internal", REINFORCE_AMOUNT / 4)
                self.reinforce_thought(node2_id, "internal", REINFORCE_AMOUNT / 4)


    def get_connection_target(self, preferred_influence=None):
        if not self.graph.nodes: return None
        
        potential_targets = []
        for node_id, data in self.graph.nodes(data=True):
            # Prefer prominent nodes
            prominence_score = data['prominence']
            # Prefer nodes matching the influence type
            influence_score = 0
            if preferred_influence and data['influences'].get(preferred_influence, 0) > 0.5:
                influence_score = data['influences'][preferred_influence]
            
            potential_targets.append((node_id, prominence_score + influence_score))
        
        if not potential_targets: return random.choice(list(self.graph.nodes())) # fallback
        
        potential_targets.sort(key=lambda x: x[1], reverse=True)
        # Select probabilistically from top N, or just take the top one
        return potential_targets[0][0] if potential_targets else random.choice(list(self.graph.nodes()))


    def update_prominence_and_prune(self):
        nodes_to_prune = []
        for node_id, data in self.graph.nodes(data=True):
            # Decay prominence if not recently reinforced
            if self.current_age - data['last_reinforced'] > TIME_STEP * 5: # Arbitrary window for reinforcement
                data['prominence'] -= PROMINENCE_DECAY_RATE * (self.current_age - data['last_reinforced'])
                data['prominence'] = max(0, data['prominence'])
                data['size'] = data['prominence'] * 500 + 100

            if data['prominence'] < PROMINENCE_PRUNE_THRESHOLD and self.current_age - data['age_created'] > 1.0: # Don't prune brand new thoughts immediately
                nodes_to_prune.append(node_id)
        
        for node_id in nodes_to_prune:
            label = self.graph.nodes[node_id]['label']
            # print(f"Age {self.current_age:.1f}: Pruning thought '{label}' ({node_id}) due to low prominence.")
            if label in self.node_labels: # remove from set of active labels
                 self.node_labels.remove(label)
            self.graph.remove_node(node_id)

    def step(self):
        if self.current_age >= MAX_AGE:
            return False # Simulation ended

        self.current_age += TIME_STEP
        self.apply_events()
        self.update_prominence_and_prune()
        return True

    def draw(self, ax):
        ax.clear()
        if not self.graph.nodes():
            ax.text(0.5, 0.5, "Mind is empty", ha='center', va='center')
            ax.set_title(f"Age: {self.current_age:.1f} years")
            return

        pos = nx.spring_layout(self.graph, k=0.5/np.sqrt(self.graph.number_of_nodes()) if self.graph.number_of_nodes() > 0 else 0.5, iterations=20)
        
        node_colors = [get_dominant_influence_color(data['influences']) for node_id, data in self.graph.nodes(data=True)]
        node_sizes = [data['size'] for node_id, data in self.graph.nodes(data=True)]
        edge_colors = [data.get('color', DEFAULT_EDGE_COLOR) for u, v, data in self.graph.edges(data=True)]
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9, ax=ax)
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, edge_color=edge_colors, alpha=0.5, width=1.5, arrows=True, arrowstyle='-|>', arrowsize=10, ax=ax)
        
        # Draw labels (only for prominent nodes to avoid clutter)
        labels = {node_id: data['label'] for node_id, data in self.graph.nodes(data=True) if data['prominence'] > 0.3}
        nx.draw_networkx_labels(self.graph, pos, labels=labels, font_size=8, ax=ax)

        ax.set_title(f"Age: {self.current_age:.1f} years | Thoughts: {self.graph.number_of_nodes()}")
        ax.axis('off')

        # Legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'{name.capitalize()} Influence',
                                     markerfacecolor=col, markersize=10) for name, col in COLOR_MAP.items()]
        ax.legend(handles=legend_elements, loc='lower left', fontsize='small')


# --- Main Simulation Loop ---
if __name__ == "__main__":
    simulation = MindSimulation()
    simulation.initialize_mind()

    fig, ax = plt.subplots(figsize=(12, 9))
    plt.ion() # Interactive mode on

    try:
        while simulation.current_age < MAX_AGE:
            if not simulation.step():
                print("Simulation ended.")
                break
            
            simulation.draw(ax)
            plt.draw()
            plt.pause(ANIMATION_PAUSE)
            
            if not plt.fignum_exists(fig.number): # Check if window was closed
                print("Plot window closed. Exiting simulation.")
                break

    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    finally:
        plt.ioff()
        if plt.fignum_exists(fig.number):
            plt.show() # Keep final plot open
        print(f"Simulation finished at age {simulation.current_age:.1f}")
        print(f"Final number of thoughts: {simulation.graph.number_of_nodes()}")
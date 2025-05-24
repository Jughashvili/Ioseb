import networkx as nx
import matplotlib.pyplot as plt
import random
import time
import numpy as np
import uuid # For unique IDs that are not just integers

# --- Configuration ---
MAX_AGE = 99
INITIAL_AGE = 0.1
TIME_STEP = 0.1
ANIMATION_PAUSE = 0.05 # Faster animation for quicker evolution
USER_INTERACTION_STEP = 30 # Ask for user input every N steps

# Prominence & Memory
INITIAL_PROMINENCE = 0.3
REINFORCE_AMOUNT = 0.15
PROMINENCE_DECAY_RATE_STM = 0.025
PROMINENCE_DECAY_RATE_LTM = 0.005
PROMINENCE_PRUNE_THRESHOLD = 0.05
MAX_PROMINENCE = 1.0
STM_TO_LTM_THRESHOLD = 0.6
REINFORCEMENTS_FOR_LTM = 3

# Event Probabilities (per time step)
SENSORY_EVENT_PROB = 0.25
SOCIAL_EVENT_PROB = 0.15
EDUCATIONAL_EVENT_PROB = 0.10
MEDIA_EVENT_PROB = 0.05 # Lower, increases with age
BIOLOGICAL_DRIVE_EVENT_PROB = 0.15
INTERNAL_REFLECTION_PROB = 0.20

# --- Colors & Types ---
# Granular Influences
INFLUENCE_TYPES = {
    "sensory": "green",
    "biological_drive": "darkgreen",
    "social_norm": "blue",
    "education": "lightblue",
    "media": "cyan",
    "cultural_belief": "deepskyblue",
    "language_construct": "cornflowerblue",
    "reflection": "purple",      # Internal abstract thought
    "association": "mediumpurple", # Connecting existing thoughts
    "deduction": "indigo",       # Forming new conclusions (simplified)
    "emotional_response": "pink" # Influence from current emotional state
}
DEFAULT_NODE_COLOR = "lightgray"
DEFAULT_EDGE_COLOR = "gray"
MIXED_INFLUENCE_COLOR = "gray"

# Node Types
NODE_TYPES_CONFIG = {
    "concept": {"shape": "o", "base_size_factor": 1.0},
    "sensory_input": {"shape": "h", "base_size_factor": 0.8}, # Hexagon
    "episodic_memory": {"shape": "s", "base_size_factor": 0.9}, # Square
    "procedural_knowledge": {"shape": "D", "base_size_factor": 1.1}, # Diamond
    "belief": {"shape": "^", "base_size_factor": 1.0}, # Triangle
    "goal": {"shape": "p", "base_size_factor": 1.2}, # Pentagon (active)
    "emotion_tag": {"shape": "*", "base_size_factor": 0.7}, # Star
    "language_unit": {"shape": "v", "base_size_factor": 0.8} # Triangle down
}

# --- Emotional State ---
EMOTIONS = {"happiness": 0.5, "fear": 0.1, "curiosity": 0.6, "sadness": 0.1, "anger": 0.1}
EMOTION_DECAY = 0.01
EMOTION_INTENSITY_THRESHOLD = 0.7 # For strong influence/tagging

# --- Potential Content Seeds (Type, Base Label, Themes, Primary Influence Source, Content Hint, Emotional Impact) ---
POTENTIAL_THOUGHT_SEEDS = {
    "sensory_input": [
        ("Warmth", ["comfort", "physical"], "sensory", {"sensation_type": "tactile"}, {"happiness": 0.1}),
        ("Loud_Noise", ["alert", "physical", "unexpected"], "sensory", {"sensation_type": "auditory"}, {"fear": 0.05, "curiosity": 0.1}),
        ("Bright_Light", ["visual", "intense"], "sensory", {"sensation_type": "visual"}, {"curiosity": 0.05}),
        ("Mother's_Scent", ["comfort", "attachment", "recognition"], "sensory", {"sensation_type": "olfactory"}, {"happiness": 0.2}),
    ],
    "biological_drive_stimulus": [
        ("Hunger_Pang", ["need", "food", "discomfort", "internal_state"], "biological_drive", {"drive_type": "hunger"}, {"sadness": 0.1, "anger":0.02}), # slight irritation
        ("Tiredness", ["need", "rest", "discomfort", "internal_state"], "biological_drive", {"drive_type": "fatigue"}, {"sadness": 0.05}),
        ("Thirst_Sensation", ["need", "water", "discomfort", "internal_state"], "biological_drive", {"drive_type": "thirst"}, {"sadness": 0.1}),
    ],
    "social_interaction": [ # Event source, generates thought
        ("Smile_From_Caregiver", ["positive", "social", "validation", "caregiver"], "social_norm", {"interaction_partner": "caregiver"}, {"happiness": 0.2}),
        ("Shared_Toy_With_Peer", ["cooperation", "social", "play", "peer"], "social_norm", {"activity": "sharing"}, {"happiness": 0.15}),
        ("Verbal_Praise", ["positive", "social", "achievement", "language"], "social_norm", {"feedback_type": "praise"}, {"happiness": 0.25}),
        ("Gentle_Correction:'No'", ["rule", "social", "guidance", "language", "limitation"], "social_norm", {"correction_type": "verbal"}, {"sadness": 0.05, "curiosity":0.05}),
    ],
    "educational_input": [ # Event source
        ("Learned_Word:'Apple'", ["language", "object", "food", "naming"], "education", {"item_learned": "word:apple", "category": "food"}, {"curiosity": 0.15}),
        ("Shown_Picture_Of_Dog", ["object_recognition", "animal", "visual_learning"], "education", {"item_learned": "concept:dog", "category":"animal"}, {"curiosity": 0.2}),
        ("Taught_Rule:'Be_Kind'", ["social_rule", "cooperation", "ethics"], "education", {"rule_type": "social_conduct"}, {"curiosity": 0.1, "happiness":0.05}), # understanding rules can be positive
        ("Learned_To_Stack_Blocks", ["skill", "motor_control", "causality", "play"], "education", {"skill_type": "fine_motor"}, {"curiosity": 0.2, "happiness":0.1}), # procedural
    ],
    "media_input": [ # Event source (more relevant later)
        ("Saw_Cartoon_Character_Fly", ["story", "fantasy", "imagination", "hero"], "media", {"media_genre": "cartoon"}, {"curiosity": 0.15, "happiness":0.05}),
        ("Heard_Catchy_Song_On_Radio", ["music", "emotion", "entertainment", "auditory_pattern"], "media", {"media_genre": "music"}, {"happiness": 0.1, "curiosity":0.05}),
    ]
}

# --- Helper Functions ---
def get_unique_id():
    return str(uuid.uuid4())

def get_node_color_and_shape(node_data):
    dominant_color = DEFAULT_NODE_COLOR
    influences_dict = node_data.get('influences', {})
    if influences_dict:
        max_inf_val = 0
        dominant_type_influence = None
        num_significant_influences = 0
        for inf_type, val in influences_dict.items():
            if val > 0.1: num_significant_influences +=1
            if val > max_inf_val:
                max_inf_val = val
                dominant_type_influence = inf_type
        if num_significant_influences > 1 and (not dominant_type_influence or max_inf_val < 0.5):
            dominant_color = MIXED_INFLUENCE_COLOR
        elif dominant_type_influence:
            dominant_color = INFLUENCE_TYPES.get(dominant_type_influence, DEFAULT_NODE_COLOR)

    shape = NODE_TYPES_CONFIG.get(node_data.get('type', 'concept'), {"shape":"o"})["shape"]
    return dominant_color, shape

def normalize_dict_values(d):
    total = sum(d.values())
    if total == 0: return d
    return {k: v / total for k, v in d.items()}

# --- Simulation Class ---
class MindSimulation:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.current_age = 0.0
        self.simulation_step = 0
        self.node_labels_set = set()
        self.emotions = EMOTIONS.copy()
        self.active_goals = {} # {goal_id: data}

    def initialize_mind(self):
        self.current_age = INITIAL_AGE
        self_id = self.add_thought(
            base_label="Self_Concept", node_type="concept",
            initial_influences={"sensory": 0.3, "reflection": 0.5, "biological_drive": 0.2},
            prominence=0.8, themes=["identity", "existence", "body_awareness"],
            memory_status="LTM", content={"description": "Core sense of self"}
        )
        sensation_hub_id = self.add_thought(
            base_label="Sensory_Processing_Hub", node_type="concept",
            initial_influences={"sensory": 0.9, "reflection": 0.1}, prominence=0.7,
            themes=["awareness", "physical_world_interface"], memory_status="LTM",
            content={"function": "Integrates sensory inputs"}
        )
        self.add_connection(sensation_hub_id, self_id, "sensory", "self_awareness_from_sensation")

        caregiver_concept_id = self.add_thought(
            base_label="Primary_Caregiver_Schema", node_type="concept",
            initial_influences={"social_norm": 0.4, "sensory": 0.4, "biological_drive":0.2}, # Drive for attachment
            prominence=0.7, themes=["attachment", "comfort", "safety", "social_bond"], memory_status="LTM",
            content={"role": "Source of care and safety"}
        )
        self.add_connection(caregiver_concept_id, self_id, "social_norm", "self_in_relation_to_caregiver")
        self.add_connection(sensation_hub_id, caregiver_concept_id, "sensory", "recognizing_caregiver_cues")
        print(f"Mind initialized at age {self.current_age:.1f} with {self.graph.number_of_nodes()} thoughts.")

    def add_thought(self, base_label, node_type, initial_influences, prominence=INITIAL_PROMINENCE,
                    themes=None, content=None, memory_status="STM", existing_node_to_connect=None,
                    connection_influence_type=None, connection_description=None):
        node_id = get_unique_id()
        label_candidate = base_label
        counter = 1
        while label_candidate in self.node_labels_set:
            label_candidate = f"{base_label}_{counter}"
            counter += 1
        final_label = label_candidate
        self.node_labels_set.add(final_label)

        node_config = NODE_TYPES_CONFIG.get(node_type, {"base_size_factor": 1.0})
        base_size = 50 * node_config["base_size_factor"]

        self.graph.add_node(
            node_id, label=final_label, type=node_type,
            influences=normalize_dict_values(initial_influences.copy()),
            prominence=min(prominence, MAX_PROMINENCE),
            age_created=self.current_age, last_reinforced=self.current_age,
            reinforcement_count=1, memory_status=memory_status,
            themes=themes if themes else [], content=content if content else {},
            size=prominence * 200 * node_config["base_size_factor"] + base_size
        )
        if existing_node_to_connect and self.graph.has_node(existing_node_to_connect) and connection_influence_type:
            self.add_connection(existing_node_to_connect, node_id, connection_influence_type, connection_description)
        return node_id

    def add_connection(self, source_id, target_id, influence_type, description="association", weight=1.0):
        if self.graph.has_node(source_id) and self.graph.has_node(target_id) and source_id != target_id:
            if not self.graph.has_edge(source_id, target_id):
                self.graph.add_edge(
                    source_id, target_id, influence_type=influence_type,
                    color=INFLUENCE_TYPES.get(influence_type, DEFAULT_EDGE_COLOR),
                    description=description, weight=weight, created_at_age=self.current_age
                )

    def reinforce_thought(self, node_id, reinforcing_influence_type=None, amount=REINFORCE_AMOUNT, context_themes=None):
        if not self.graph.has_node(node_id): return
        node = self.graph.nodes[node_id]
        bonus = 0
        if context_themes and node.get('themes'):
            shared_themes = set(context_themes).intersection(set(node['themes']))
            bonus = len(shared_themes) * (amount / 2)

        node['prominence'] = min(MAX_PROMINENCE, node['prominence'] + amount + bonus)
        node['last_reinforced'] = self.current_age
        node['reinforcement_count'] += 1
        
        node_config = NODE_TYPES_CONFIG.get(node['type'], {"base_size_factor": 1.0})
        base_size = 50 * node_config["base_size_factor"]
        node['size'] = node['prominence'] * 200 * node_config["base_size_factor"] + base_size


        if reinforcing_influence_type:
            node['influences'][reinforcing_influence_type] = node['influences'].get(reinforcing_influence_type, 0) + 0.1
            node['influences'] = normalize_dict_values(node['influences'])
        
        if node['memory_status'] == "STM" and \
           (node['prominence'] > STM_TO_LTM_THRESHOLD or node['reinforcement_count'] > REINFORCEMENTS_FOR_LTM):
            node['memory_status'] = "LTM"

        # If a goal is reinforced heavily, consider it "achieved" or "progressed"
        if node['type'] == 'goal' and node['prominence'] > 0.8:
             if node_id in self.active_goals:
                 self.active_goals[node_id]['status'] = 'achieved_recently'
                 node['content']['status'] = 'achieved_recently'
                 # print(f"Goal '{node['label']}' marked as achieved!")
                 # This could trigger broader reinforcement of contributing thoughts/skills

    def update_emotional_state(self, event_emotional_impact=None):
        if event_emotional_impact:
            for emotion, change in event_emotional_impact.items():
                self.emotions[emotion] = min(1.0, max(0.0, self.emotions[emotion] + change))
                # If strong emotion, create an emotion_tag node
                if abs(change) > 0.1 and self.emotions[emotion] > EMOTION_INTENSITY_THRESHOLD * 0.8: # If significant change leading to strong emotion
                    self.add_thought(
                        base_label=f"Felt_{emotion.capitalize()}", node_type="emotion_tag",
                        initial_influences={"emotional_response": 0.9, "reflection":0.1}, prominence=0.4,
                        themes=[emotion, "affective_state", "internal_experience"],
                        content={"intensity": self.emotions[emotion], "trigger_event_themes": event_emotional_impact.get("trigger_themes", [])}
                    )

        for emotion in self.emotions: # Decay
            target_baseline = 0.5 if emotion == "happiness" or emotion == "curiosity" else 0.1
            if self.emotions[emotion] > target_baseline:
                self.emotions[emotion] = max(target_baseline, self.emotions[emotion] - EMOTION_DECAY)
            else:
                self.emotions[emotion] = min(target_baseline, self.emotions[emotion] + EMOTION_DECAY / 2)

    def generate_event(self):
        # Weighted choice of event type, can be age-dependent
        event_type_probs = {
            "sensory_input": SENSORY_EVENT_PROB,
            "biological_drive_stimulus": BIOLOGICAL_DRIVE_EVENT_PROB,
            "social_interaction": SOCIAL_EVENT_PROB * (1 + self.current_age/20), # Increases with age
            "educational_input": EDUCATIONAL_EVENT_PROB * (1 + self.current_age/10), # Increases with age
            "media_input": MEDIA_EVENT_PROB * (self.current_age/5 if self.current_age > 3 else 0) # Starts later
        }
        event_types = list(event_type_probs.keys())
        weights = [event_type_probs[et] for et in event_types]
        
        if sum(weights) == 0 or random.random() > sum(weights): # Chance of no external event
             return None

        chosen_event_category = random.choices(event_types, weights=weights, k=1)[0]
        
        if not POTENTIAL_THOUGHT_SEEDS.get(chosen_event_category): return None # Should not happen if configured well

        label_base, themes, prim_inf, content_hint, emo_impact = random.choice(POTENTIAL_THOUGHT_SEEDS[chosen_event_category])
        
        # Add current strong emotion to themes of event, influencing processing
        current_strong_emotion = [e for e,v in self.emotions.items() if v > EMOTION_INTENSITY_THRESHOLD]
        if current_strong_emotion:
            themes = themes + [f"felt_{current_strong_emotion[0]}"]

        return {
            "category": chosen_event_category, "source_influence": prim_inf,
            "label_base": label_base, "themes": themes, "content": content_hint,
            "emotional_impact": emo_impact.copy() # Ensure mutable copy
        }

    def process_event(self, event_data):
        if not event_data: return

        self.update_emotional_state(event_data.get("emotional_impact")) # Update emotion first

        node_type_hint = "concept" # Default
        if event_data["category"] == "sensory_input": node_type_hint = "sensory_input"
        elif event_data["category"] == "educational_input" and "skill" in event_data["themes"]: node_type_hint = "procedural_knowledge"
        elif event_data["category"] == "educational_input" and "word" in event_data["content"].get("item_learned","").lower(): node_type_hint = "language_unit"
        elif event_data["category"] == "educational_input" and "rule" in event_data["themes"]: node_type_hint = "belief" # or rule


        new_thought_influences = {event_data["source_influence"]: 0.6}
        # Add influence from current dominant emotion
        dominant_emotion = max(self.emotions, key=self.emotions.get)
        if self.emotions[dominant_emotion] > EMOTION_INTENSITY_THRESHOLD * 0.7:
            new_thought_influences["emotional_response"] = new_thought_influences.get("emotional_response", 0) + 0.4 * self.emotions[dominant_emotion]
        new_thought_influences = normalize_dict_values(new_thought_influences)

        target_node_id = self.find_related_thought(event_data["themes"], event_data["source_influence"])
        
        new_node_id = self.add_thought(
            base_label=event_data["label_base"], node_type=node_type_hint,
            initial_influences=new_thought_influences,
            prominence=INITIAL_PROMINENCE + (0.1 if target_node_id else 0),
            themes=event_data["themes"], content=event_data["content"],
            existing_node_to_connect=target_node_id,
            connection_influence_type=event_data["source_influence"],
            connection_description=f"triggered_by_{event_data['category']}"
        )

        if target_node_id:
            self.reinforce_thought(target_node_id, event_data["source_influence"], REINFORCE_AMOUNT / 2, context_themes=event_data["themes"])
        
        # Broader contextual reinforcement for nodes sharing themes
        for node_id, data in list(self.graph.nodes(data=True)): # Iterate over copy
            if node_id == new_node_id or node_id == target_node_id: continue
            if data.get('themes'):
                shared_themes = set(event_data["themes"]).intersection(set(data['themes']))
                if len(shared_themes) > 0:
                    self.reinforce_thought(node_id, event_data["source_influence"], REINFORCE_AMOUNT / (2 + len(shared_themes)), context_themes=event_data["themes"])

    def find_related_thought(self, themes, preferred_influence_type):
        if not self.graph.nodes: return None
        candidates = []
        for node_id, data in self.graph.nodes(data=True):
            score = 0
            if data.get('themes') and themes:
                shared = set(themes).intersection(set(data['themes']))
                score += len(shared) * 2.5 # Theme match is important
            if data['influences'].get(preferred_influence_type, 0) > 0.2: # Check primary influence match
                score += data['influences'][preferred_influence_type] * 1.5
            score += data['prominence'] # General prominence
            if data['memory_status'] == 'LTM': score += 0.5 # Prefer LTM
            # Recency bonus
            age_factor = max(0.1, (self.current_age - data['last_reinforced']))
            score += 0.5 / age_factor

            if score > 0.5 : candidates.append((node_id, score)) # Min score to be considered
        
        if not candidates: return None
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0] if candidates[0][1] > 1.5 else None # Need a reasonably good match


    def internal_processes(self):
        # 1. Reflection, Association, Deduction
        if random.random() < INTERNAL_REFLECTION_PROB and self.graph.number_of_nodes() >= 2:
            # Prefer LTM, prominent thoughts for reflection
            potential_sources = [nid for nid, data in self.graph.nodes(data=True) if data['prominence'] > 0.4 and data['memory_status'] == 'LTM']
            if len(potential_sources) < 2: return

            node1_id, node2_id = random.sample(potential_sources, 2)
            node1_data, node2_data = self.graph.nodes[node1_id], self.graph.nodes[node2_id]

            # Simple association if not already strongly connected
            if not self.graph.has_edge(node1_id, node2_id) and not self.graph.has_edge(node2_id, node1_id):
                shared_themes_count = len(set(node1_data.get('themes', [])).intersection(set(node2_data.get('themes', []))))
                if shared_themes_count > 0 or random.random() < 0.2: # Connect if shared themes or randomly
                    conn_influence = "association"
                    if node1_data['type'] == 'concept' and node2_data['type'] == 'concept': conn_influence = "reflection"
                    
                    self.add_connection(node1_id, node2_id, conn_influence, f"internally_linked")
                    self.reinforce_thought(node1_id, conn_influence, REINFORCE_AMOUNT / 4, context_themes=node2_data.get('themes'))
                    self.reinforce_thought(node2_id, conn_influence, REINFORCE_AMOUNT / 4, context_themes=node1_data.get('themes'))

                    # Simplified "Deduction" / "Insight" Creation
                    if random.random() < 0.15 and shared_themes_count >=1 : # More likely if themes overlap
                        new_label = f"Insight:[{node1_data['label'][:7]}+{node2_data['label'][:7]}]"
                        new_themes = list(set(node1_data.get('themes', [])).union(set(node2_data.get('themes', [])))) + ["derived_insight", "abstract"]
                        
                        # Mix influences, boost 'deduction' or 'reflection'
                        new_influences = {k: (node1_data['influences'].get(k,0) + node2_data['influences'].get(k,0))/2 for k in set(node1_data['influences']) | set(node2_data['influences'])}
                        new_influences["deduction"] = new_influences.get("deduction", 0) + 0.4
                        new_influences = normalize_dict_values(new_influences)
                        
                        insight_content = {"derived_from": [node1_data['label'], node2_data['label']], "common_themes": list(set(node1_data.get('themes', [])).intersection(set(node2_data.get('themes', []))))}

                        new_insight_id = self.add_thought(base_label=new_label, node_type="concept",
                                                       initial_influences=new_influences, themes=new_themes,
                                                       prominence=(node1_data['prominence'] + node2_data['prominence'])/3,
                                                       content=insight_content, memory_status="STM") # Insights start as STM
                        self.add_connection(node1_id, new_insight_id, "reflection", "contributed_to_insight")
                        self.add_connection(node2_id, new_insight_id, "reflection", "contributed_to_insight")
                        self.add_connection(new_insight_id, node1_id, "association", "insight_relates_back") # Connect back too
                        self.add_connection(new_insight_id, node2_id, "association", "insight_relates_back")

        # 2. Goal Management
        # Check active goals
        for goal_id, goal_data_copy in list(self.active_goals.items()): # Iterate over copy
            if not self.graph.has_node(goal_id):
                del self.active_goals[goal_id]
                continue
            
            goal_node = self.graph.nodes[goal_id]
            if goal_node['content'].get('status') == 'achieved_recently':
                # print(f"Processing achieved goal: {goal_node['label']}")
                # Reinforce path leading to goal (very simplified - needs proper path tracking)
                # For now, reinforce thoughts with shared themes that are procedural or concepts
                goal_themes = set(goal_node.get('themes',[]))
                for nid, ndata in list(self.graph.nodes(data=True)):
                    if ndata.get('type') in ['procedural_knowledge', 'concept'] and \
                       len(goal_themes.intersection(set(ndata.get('themes',[])))) > 0 and \
                       ndata['last_reinforced'] > goal_node['age_created']: # Active during goal pursuit
                        self.reinforce_thought(nid, "reflection", REINFORCE_AMOUNT * 0.5, context_themes=list(goal_themes))
                goal_node['content']['status'] = 'archived'
                goal_node['prominence'] *= 0.7 # Fade a bit after achievement, but still LTM
                if goal_id in self.active_goals: del self.active_goals[goal_id]


        # New Goal Formation from Drives or Strong Desires
        if len(self.active_goals) < 3: # Limit concurrent active goals
            for node_id, data in self.graph.nodes(data=True):
                is_drive_node = data['type'] == 'sensory_input' and "discomfort" in data.get('themes',[]) and "need" in data.get('themes',[])
                is_strong_desire = data['type'] == 'concept' and data['prominence'] > 0.7 and "desire" in data.get('themes',[]) # Future: desire theme
                
                if (is_drive_node and data['prominence'] > 0.6) or is_strong_desire:
                    # Check if similar goal already active
                    potential_goal_theme = None
                    if "food" in data['themes']: potential_goal_theme = "obtain_food"
                    elif "rest" in data['themes']: potential_goal_theme = "obtain_rest"
                    elif "water" in data['themes']: potential_goal_theme = "obtain_water"
                    elif "comfort" in data['themes'] and "discomfort" in data['themes']: potential_goal_theme = "seek_comfort"
                    # Add more goal themes based on desire nodes...

                    if potential_goal_theme:
                        already_active = any(g['content'].get('core_theme') == potential_goal_theme for g in self.active_goals.values())
                        if not already_active:
                            goal_label = f"Goal:{potential_goal_theme.replace('_',' ').title()}"
                            goal_id = self.add_thought(base_label=goal_label, node_type="goal",
                                             initial_influences={"biological_drive":0.5, "reflection":0.5} if is_drive_node else {"reflection":0.7, "emotional_response":0.3},
                                             prominence=0.7, themes=["goal", potential_goal_theme, "motivation", "active_pursuit"],
                                             content={"status":"active", "core_theme": potential_goal_theme, "urgency": data['prominence']})
                            self.add_connection(node_id, goal_id, data['influences'] if isinstance(data['influences'],str) else "biological_drive", f"triggered_goal:{potential_goal_theme}")
                            self.active_goals[goal_id] = self.graph.nodes[goal_id].copy() # Store a copy
                            # print(f"Age {self.current_age:.1f}: Formed Goal '{goal_label}'")
                            break # One new goal per step


    def update_prominence_and_prune(self):
        nodes_to_prune = []
        for node_id, data in list(self.graph.nodes(data=True)):
            decay_rate = PROMINENCE_DECAY_RATE_STM if data['memory_status'] == "STM" else PROMINENCE_DECAY_RATE_LTM
            
            if data['memory_status'] == "STM" and self.emotions["curiosity"] > EMOTION_INTENSITY_THRESHOLD:
                decay_rate *= 0.3 # High curiosity protects STM thoughts significantly
            if data['memory_status'] == "LTM" and (self.emotions["sadness"] > 0.6 or (self.emotions["happiness"] < 0.2 and self.emotions["curiosity"] < 0.2)):
                 decay_rate *= 1.8 # Sadness or apathy/boredom accelerates LTM decay

            # Goals decay slower if active
            if data['type'] == 'goal' and data.get('content',{}).get('status') == 'active':
                decay_rate *= 0.5

            time_since_reinforced = self.current_age - data['last_reinforced']
            if time_since_reinforced > TIME_STEP * 1.5: # Decay if not very recently touched
                 data['prominence'] -= decay_rate * (time_since_reinforced / (TIME_STEP * 10)) # Slower base decay scaled by time
                 data['prominence'] = max(0, data['prominence'])
                 
                 node_config = NODE_TYPES_CONFIG.get(data['type'], {"base_size_factor": 1.0})
                 base_size = 50 * node_config["base_size_factor"]
                 data['size'] = data['prominence'] * 200 * node_config["base_size_factor"] + base_size


            prune_age_threshold = (10 if data['memory_status'] == 'LTM' else 2.5) # LTMs persist longer
            if data['type'] == 'goal': prune_age_threshold = max(prune_age_threshold, 15) # Goals are important

            if data['prominence'] < PROMINENCE_PRUNE_THRESHOLD and self.current_age - data['age_created'] > prune_age_threshold :
                nodes_to_prune.append(node_id)
        
        for node_id in nodes_to_prune:
            if self.graph.has_node(node_id):
                label = self.graph.nodes[node_id]['label']
                if label in self.node_labels_set: self.node_labels_set.remove(label)
                if node_id in self.active_goals: del self.active_goals[node_id]
                self.graph.remove_node(node_id)

    def step(self):
        if self.current_age >= MAX_AGE: return False
        self.current_age += TIME_STEP
        self.simulation_step += 1

        event = self.generate_event()
        self.process_event(event)
        self.internal_processes()
        self.update_emotional_state() # General decay/normalization after all event processing
        self.update_prominence_and_prune()
        return True

    def draw(self, ax):
        ax.clear()
        if not self.graph.nodes():
            ax.text(0.5, 0.5, "Mind is empty", ha='center', va='center')
            ax.set_title(f"Age: {self.current_age:.1f} years")
            return

        # Use a layout that is more stable if possible, or recompute less often if slow
        # For dynamic graphs, spring_layout is okay but can be jumpy.
        # pos = nx.nx_agraph.graphviz_layout(self.graph, prog="neato") # if pygraphviz installed
        pos = nx.spring_layout(self.graph, k=1.2/np.sqrt(self.graph.number_of_nodes()) if self.graph.number_of_nodes() > 0 else 0.5, iterations=20, seed=42) # Seed for some stability
        
        # Group nodes by shape for efficient drawing with different markers
        nodes_by_shape = {}
        for node_id, data in self.graph.nodes(data=True):
            color, shape_marker = get_node_color_and_shape(data)
            if shape_marker not in nodes_by_shape:
                nodes_by_shape[shape_marker] = {"nodelist": [], "colors": [], "sizes": []}
            nodes_by_shape[shape_marker]["nodelist"].append(node_id)
            nodes_by_shape[shape_marker]["colors"].append(color)
            nodes_by_shape[shape_marker]["sizes"].append(data['size'])

        for shape_marker, node_group in nodes_by_shape.items():
            nx.draw_networkx_nodes(self.graph, pos, nodelist=node_group["nodelist"], node_shape=shape_marker,
                                   node_color=node_group["colors"], node_size=node_group["sizes"], alpha=0.9, ax=ax)

        edge_colors = [data.get('color', DEFAULT_EDGE_COLOR) for u, v, data in self.graph.edges(data=True)]
        edge_widths = [max(0.5, data.get('weight',1.0) * 1.2) for u,v,data in self.graph.edges(data=True)]
        nx.draw_networkx_edges(self.graph, pos, edge_color=edge_colors, alpha=0.35, width=edge_widths, 
                               arrows=True, arrowstyle='-|>', arrowsize=8, connectionstyle='arc3,rad=0.05', ax=ax)
        
        labels_to_draw = {nid: data['label'] for nid, data in self.graph.nodes(data=True) if data['prominence'] > 0.4 or data['type'] == 'goal'}
        nx.draw_networkx_labels(self.graph, pos, labels=labels_to_draw, font_size=6, ax=ax, font_weight='normal')

        num_stm = sum(1 for _,d in self.graph.nodes(data=True) if d['memory_status']=='STM')
        num_ltm = self.graph.number_of_nodes() - num_stm
        title_text = (f"Age: {self.current_age:.1f} | Thoughts: {self.graph.number_of_nodes()} "
                      f"(LTM: {num_ltm}, STM: {num_stm}) | Active Goals: {len(self.active_goals)}")
        ax.set_title(title_text, fontsize=10)
        ax.axis('off')

        # Legend for influence types
        influence_legend = [plt.Line2D([0], [0], marker='o', color='w', label=name.replace("_", " ").title(),
                                     markerfacecolor=col, markersize=6) for name, col in INFLUENCE_TYPES.items()]
        # Legend for node types (shapes)
        type_legend = [plt.Line2D([0], [0], marker=cfg["shape"], color='w', label=ntype.replace("_", " ").title(),
                                   markerfacecolor='gray', markeredgecolor='black', markersize=8, linestyle='None')
                       for ntype, cfg in NODE_TYPES_CONFIG.items()]
        
        first_legend = ax.legend(handles=influence_legend, loc='lower left', fontsize='xx-small', ncol=2, title="Influences", title_fontsize='x-small')
        ax.add_artist(first_legend)
        ax.legend(handles=type_legend, loc='lower right', fontsize='xx-small', ncol=1, title="Node Types", title_fontsize='x-small')


        emotion_text = "Emotions:\n" + "\n".join([f"{e.capitalize()[:5]}: {self.emotions[e]:.2f}" for e in self.emotions])
        ax.text(0.01, 0.99, emotion_text, transform=ax.transAxes, fontsize=6,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.2', fc='ivory', alpha=0.7))

    def handle_user_input(self):
        print("\n--- User Interaction ---")
        print(f"Current Age: {self.current_age:.1f}, Sim Step: {self.simulation_step}")
        action = input("Inject thought (t), event (e), reinforce (r), info (i), or continue (c)? [c]: ").lower()

        if action == 't':
            label = input("Thought label: ")
            ttype = input(f"Thought type {list(NODE_TYPES_CONFIG.keys())}: ")
            if ttype not in NODE_TYPES_CONFIG: ttype = "concept"
            influence_src = input(f"Primary influence source {list(INFLUENCE_TYPES.keys())}: ")
            if influence_src not in INFLUENCE_TYPES: influence_src = "reflection"
            themes_str = input("Themes (comma-separated): ")
            themes = [t.strip() for t in themes_str.split(',') if t.strip()]
            
            self.add_thought(base_label=label, node_type=ttype, initial_influences={influence_src: 1.0}, themes=themes)
            print("User thought injected.")
        elif action == 'e':
            print("Simulating a generic 'user_defined' event.")
            event_label = input("Event description (becomes thought label): ")
            event_themes_str = input("Event themes (comma-separated): ")
            event_themes = [t.strip() for t in event_themes_str.split(',') if t.strip()]
            event_influence = input(f"Event influence source {list(INFLUENCE_TYPES.keys())}: ")
            if event_influence not in INFLUENCE_TYPES: event_influence = "social_norm"
            
            user_event = {
                "category": "user_defined", "source_influence": event_influence, "label_base": event_label,
                "themes": event_themes, "content": {"details": "user-defined event", "user_specified": True},
                "emotional_impact": {"curiosity": 0.15, "happiness":0.05}
            }
            self.process_event(user_event)
            print("User event processed.")
        elif action == 'r':
            if not self.graph.nodes: print("No thoughts to reinforce."); return
            print("Available thoughts (idx, ID prefix, Prominence, Label):")
            nodes_for_reinforce = list(self.graph.nodes(data=True))
            for i, (node_id, data) in enumerate(nodes_for_reinforce):
                print(f"  {i}. {node_id[:6]} (P:{data['prominence']:.2f}) {data['label']}")
            try:
                choice_idx = int(input("Enter number of thought to reinforce: "))
                if 0 <= choice_idx < len(nodes_for_reinforce):
                    node_to_reinforce_id, data = nodes_for_reinforce[choice_idx]
                    self.reinforce_thought(node_to_reinforce_id, "reflection", REINFORCE_AMOUNT * 2.5, context_themes=data.get('themes'))
                    print(f"Reinforced '{self.graph.nodes[node_to_reinforce_id]['label']}'.")
                else: print("Invalid choice.")
            except ValueError: print("Invalid input.")
        elif action == 'i':
            node_id_query = input("Enter node ID prefix to get info (or leave blank for general): ")
            found = False
            if node_id_query:
                for node_id, data in self.graph.nodes(data=True):
                    if node_id.startswith(node_id_query):
                        print(f"\n--- Node Info: {node_id} ---")
                        for key, val in data.items():
                            if isinstance(val, dict):
                                print(f"  {key}:")
                                for k2, v2 in val.items(): print(f"    {k2}: {v2}")
                            elif isinstance(val, list):
                                print(f"  {key}: {', '.join(map(str,val)) if val else '[]'}")
                            else:
                                print(f"  {key}: {val}")
                        found = True
                        break
                if not found: print("Node not found.")
            else:
                 print(f"Total Thoughts: {self.graph.number_of_nodes()}, STM: {sum(1 for _,d in self.graph.nodes(data=True) if d['memory_status']=='STM')}")
                 print(f"Active Goals: {len(self.active_goals)}")
                 for gid, gdata in self.active_goals.items(): print(f"  - {self.graph.nodes[gid]['label']} (Urg: {gdata['content'].get('urgency',0):.2f})")
        else:
            print("Continuing simulation.")

# --- Main Simulation Loop ---
if __name__ == "__main__":
    simulation = MindSimulation()
    simulation.initialize_mind()

    fig, ax = plt.subplots(figsize=(18, 12))
    plt.ion() 

    try:
        while simulation.current_age < MAX_AGE:
            if not simulation.step():
                print("Simulation ended by max age or other condition.")
                break
            
            # Draw less frequently for performance if needed
            if simulation.simulation_step % 1 == 0: # Draw every step
                simulation.draw(ax)
                plt.draw()
            
            if simulation.simulation_step % USER_INTERACTION_STEP == 0 and simulation.simulation_step > 0:
                plt.pause(0.01) 
                simulation.handle_user_input()
                simulation.draw(ax) # Redraw if user made changes
                plt.draw()

            plt.pause(ANIMATION_PAUSE)
            
            if not plt.fignum_exists(fig.number):
                print("Plot window closed. Exiting simulation.")
                break
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    finally:
        plt.ioff()
        if plt.fignum_exists(fig.number):
            print("Displaying final state of the mind. Close plot window to exit.")
            simulation.draw(ax)
            plt.show() 
        print(f"Simulation finished at age {simulation.current_age:.1f}")
        print(f"Final number of thoughts: {self.graph.number_of_nodes()}")
        print(f"Final emotional state: {simulation.emotions}")
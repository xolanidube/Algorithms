import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import scipy.sparse as sparse
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Interaction:
    """Represents a single user interaction with an item"""
    item_id: int
    timestamp: datetime
    context: Dict[str, Any]
    rating: float

@dataclass
class Context:
    """Represents the context of a user interaction"""
    time_of_day: int  # Hour of day (0-23)
    day_of_week: int  # (0-6)
    device_type: str
    location: str
    custom_features: Dict[str, Any]

class ContextWindow:
    """Implements adaptive context window with dynamic sizing"""
    
    def __init__(self, initial_size: int = 100, min_size: int = 20, max_size: int = 1000):
        self.size = initial_size
        self.min_size = min_size
        self.max_size = max_size
        self.interactions: List[Interaction] = []
        self.context_vectors: List[np.ndarray] = []
        self.stability_score = 0.0
        
    def add_interaction(self, interaction: Interaction, context_vector: np.ndarray):
        """Add new interaction and maintain window size"""
        self.interactions.append(interaction)
        self.context_vectors.append(context_vector)
        
        # Trim if exceeding current window size
        if len(self.interactions) > self.size:
            self.interactions = self.interactions[-self.size:]
            self.context_vectors = self.context_vectors[-self.size:]
    
    def calculate_stability(self) -> float:
        """Calculate preference stability score within the window"""
        if len(self.interactions) < 2:
            return 1.0
            
        ratings = np.array([i.rating for i in self.interactions])
        stability = 1.0 - np.std(ratings) / (np.max(ratings) - np.min(ratings) + 1e-6)
        self.stability_score = stability
        return stability

class UserProfile:
    """Manages user preference data and context history"""
    
    def __init__(self, user_id: int, embedding_dim: int = 128):
        self.user_id = user_id
        self.embedding_dim = embedding_dim
        self.short_term_memory = ContextWindow()
        self.long_term_memory: List[Interaction] = []
        self.preference_vector = np.zeros(embedding_dim)
        self.interaction_count = 0
        
    def update_preference_vector(self, context_vector: np.ndarray, interaction: Interaction):
        """Update user preference vector based on new interaction"""
        learning_rate = 0.1
        item_rating_weight = interaction.rating / 5.0  # Normalize rating to [0,1]
        
        # Update preference vector using weighted average
        self.preference_vector = (1 - learning_rate) * self.preference_vector + \
                               learning_rate * (context_vector * item_rating_weight)
        
        self.interaction_count += 1

class ACARA:
    """Main implementation of the Adaptive Context-Aware Recommendation Algorithm"""
    
    def __init__(self, 
                 embedding_dim: int = 128,
                 privacy_threshold: float = 0.8,
                 num_hash_functions: int = 10):
        
        self.embedding_dim = embedding_dim
        self.privacy_threshold = privacy_threshold
        self.num_hash_functions = num_hash_functions
        
        # Initialize projection matrices for feature hashing
        self.projection_matrix = self._initialize_projections()
        
        # User profiles storage
        self.user_profiles: Dict[int, UserProfile] = {}
        
        # Item embeddings storage
        self.item_embeddings: Dict[int, np.ndarray] = {}
        
        logger.info(f"Initialized ACARA with embedding_dim={embedding_dim}")
    
    def _initialize_projections(self) -> np.ndarray:
        """Initialize random projection matrix for feature hashing"""
        return np.random.normal(0, 1, (self.embedding_dim, self.embedding_dim))
    
    def _generate_context_vector(self, context: Context) -> np.ndarray:
        """Generate privacy-preserving context embedding"""
        
        # Extract numerical features
        base_features = np.array([
            context.time_of_day / 24.0,  # Normalize time
            context.day_of_week / 6.0,   # Normalize day
            hash(context.device_type) % self.embedding_dim,
            hash(context.location) % self.embedding_dim
        ])
        
        # Add custom features
        custom_features = np.array([
            hash(f"{k}:{v}") % self.embedding_dim 
            for k, v in context.custom_features.items()
        ])
        
        # Combine features
        combined = np.concatenate([base_features, custom_features])
        
        # Apply feature hashing for privacy preservation
        hashed_vector = np.zeros(self.embedding_dim)
        for i in range(self.num_hash_functions):
            hash_idx = hash(f"{i}:{combined.tobytes()}") % self.embedding_dim
            hashed_vector[hash_idx] = 1.0
        
        # Project to final embedding space
        context_vector = np.dot(hashed_vector, self.projection_matrix)
        
        # Normalize
        return context_vector / (np.linalg.norm(context_vector) + 1e-6)
    
    def _get_or_create_profile(self, user_id: int) -> UserProfile:
        """Get existing user profile or create new one"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id, self.embedding_dim)
        return self.user_profiles[user_id]
    
    def _adapt_window_size(self, profile: UserProfile):
        """Adapt context window size based on stability and interaction rate"""
        window = profile.short_term_memory
        stability = window.calculate_stability()
        
        # Calculate interaction rate (interactions per hour)
        if len(window.interactions) < 2:
            return
            
        time_span = (window.interactions[-1].timestamp - 
                    window.interactions[0].timestamp).total_seconds() / 3600
        interaction_rate = len(window.interactions) / (time_span + 1e-6)
        
        # Adjust window size based on stability and interaction rate
        if stability > 0.8 and interaction_rate < 0.2:
            new_size = int(window.size * 0.8)
        elif stability < 0.3 or interaction_rate > 0.8:
            new_size = int(window.size * 1.2)
        else:
            return
            
        # Apply size constraints
        window.size = max(window.min_size, min(window.max_size, new_size))
        logger.debug(f"Adjusted window size to {window.size} for user {profile.user_id}")
    
    def update(self, 
               user_id: int, 
               interaction: Interaction) -> None:
        """Update user profile with new interaction"""
        
        # Generate context vector
        context_vector = self._generate_context_vector(interaction.context)
        
        # Get or create user profile
        profile = self._get_or_create_profile(user_id)
        
        # Update memories
        profile.short_term_memory.add_interaction(interaction, context_vector)
        if len(profile.long_term_memory) < 1000:  # Limit long-term memory size
            profile.long_term_memory.append(interaction)
        
        # Update preference vector
        profile.update_preference_vector(context_vector, interaction)
        
        # Adapt window size
        self._adapt_window_size(profile)
        
        logger.debug(f"Updated profile for user {user_id}")
    
    def _score_candidates(self, 
                         user_id: int, 
                         candidate_items: List[int],
                         context: Context) -> List[Tuple[int, float]]:
        """Score candidate items for recommendation"""
        
        profile = self._get_or_create_profile(user_id)
        context_vector = self._generate_context_vector(context)
        
        scores = []
        for item_id in candidate_items:
            if item_id not in self.item_embeddings:
                continue
                
            item_embedding = self.item_embeddings[item_id]
            
            # Compute score using both short-term and long-term preferences
            short_term_score = np.dot(context_vector, item_embedding)
            long_term_score = np.dot(profile.preference_vector, item_embedding)
            
            # Weighted combination of scores
            stability = profile.short_term_memory.stability_score
            final_score = stability * long_term_score + (1 - stability) * short_term_score
            
            scores.append((item_id, final_score))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)
    
    def recommend(self, 
                 user_id: int, 
                 context: Context, 
                 num_recommendations: int = 10) -> List[int]:
        """Generate recommendations for user in current context"""
        
        # Get candidate items (in practice, this would use a more sophisticated retrieval system)
        candidate_items = list(self.item_embeddings.keys())
        
        # Score candidates
        scored_items = self._score_candidates(user_id, candidate_items, context)
        
        # Return top-N recommendations
        return [item_id for item_id, _ in scored_items[:num_recommendations]]
    
    def add_item(self, item_id: int, embedding: np.ndarray):
        """Add or update item embedding"""
        normalized_embedding = embedding / (np.linalg.norm(embedding) + 1e-6)
        self.item_embeddings[item_id] = normalized_embedding

# Example usage and testing code
def run_example():
    # Initialize ACARA
    acara = ACARA(embedding_dim=128)
    
    # Create sample item embeddings
    for item_id in range(100):
        embedding = np.random.normal(0, 1, 128)
        acara.add_item(item_id, embedding)
    
    # Create sample user interactions
    user_id = 1
    base_context = Context(
        time_of_day=14,
        day_of_week=2,
        device_type="mobile",
        location="home",
        custom_features={"weather": "sunny"}
    )
    
    # Simulate interactions
    for i in range(10):
        interaction = Interaction(
            item_id=i,
            timestamp=datetime.now(),
            context=base_context,
            rating=4.0 + np.random.normal(0, 0.5)
        )
        acara.update(user_id, interaction)
    
    # Get recommendations
    recommendations = acara.recommend(user_id, base_context, num_recommendations=5)
    logger.info(f"Top 5 recommendations: {recommendations}")

if __name__ == "__main__":
    run_example()
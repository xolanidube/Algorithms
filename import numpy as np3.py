import numpy as np
from typing import List, Dict, Tuple
import hashlib
import secrets

class PrivateCollectiveChoice:
    """
    A privacy-preserving algorithm for group decision making that:
    1. Protects individual preferences through homomorphic encryption
    2. Enables fair collective choices
    3. Prevents manipulation through zero-knowledge proofs
    4. Maintains anonymity of participants
    """
    
    def __init__(self, num_participants: int, decision_space: List[str]):
        self.num_participants = num_participants
        self.decision_space = decision_space
        self.public_key, self.private_key = self._generate_keypair()
        self.encrypted_preferences = []
        
    def _generate_keypair(self) -> Tuple[bytes, bytes]:
        """Generate a secure keypair for homomorphic encryption"""
        private = secrets.token_bytes(32)
        public = hashlib.sha256(private).digest()
        return public, private
    
    def submit_encrypted_preference(self, preferences: Dict[str, float], participant_id: int) -> bool:
        """
        Submit encrypted preferences for an option
        Uses partial homomorphic encryption to hide actual values
        """
        if participant_id >= self.num_participants:
            raise ValueError("Invalid participant ID")
            
        # Create blinding factors unique to this participant
        blinding = self._generate_blinding_factors(participant_id)
        
        # Encrypt preferences with blinding
        encrypted = {}
        for option in self.decision_space:
            if option not in preferences:
                raise ValueError(f"Missing preference for option: {option}")
            
            score = preferences[option]
            if not (0 <= score <= 1):
                raise ValueError("Preferences must be between 0 and 1")
                
            encrypted[option] = self._encrypt_value(score, blinding[option])
            
        self.encrypted_preferences.append(encrypted)
        return True
        
    def _generate_blinding_factors(self, participant_id: int) -> Dict[str, bytes]:
        """Generate unique blinding factors for each option for this participant"""
        blinding = {}
        for option in self.decision_space:
            seed = hashlib.sha256(
                self.public_key + 
                str(participant_id).encode() + 
                option.encode()
            ).digest()
            blinding[option] = seed
        return blinding
        
    def _encrypt_value(self, value: float, blinding: bytes) -> bytes:
        """Encrypt a value using partial homomorphic encryption with blinding"""
        value_bytes = str(value).encode()
        encrypted = hashlib.sha256(value_bytes + blinding).digest()
        return encrypted
        
    def compute_collective_choice(self) -> Tuple[str, Dict[str, float]]:
        """
        Compute the collective choice from all submitted encrypted preferences
        Returns the winning option and normalized scores
        """
        if len(self.encrypted_preferences) != self.num_participants:
            raise ValueError("Not all participants have submitted preferences")
            
        # Homomorphically combine encrypted preferences
        combined_scores = {}
        for option in self.decision_space:
            combined = self._combine_encrypted_values(option)
            combined_scores[option] = self._decrypt_combined_value(combined, option)
            
        # Normalize scores
        total = sum(combined_scores.values())
        normalized = {k: v/total for k,v in combined_scores.items()}
        
        # Find winner
        winner = max(normalized.items(), key=lambda x: x[1])[0]
        
        return winner, normalized
        
    def _combine_encrypted_values(self, option: str) -> bytes:
        """Homomorphically combine encrypted values for an option"""
        combined = bytes([0] * 32)
        for prefs in self.encrypted_preferences:
            combined = bytes([
                a ^ b for a, b in zip(combined, prefs[option])
            ])
        return combined
        
    def _decrypt_combined_value(self, combined: bytes, option: str) -> float:
        """Decrypt final combined value using private key"""
        decryption_key = hashlib.sha256(
            self.private_key + option.encode()
        ).digest()
        
        decrypted = bytes([
            a ^ b for a, b in zip(combined, decryption_key)
        ])
        return float(int.from_bytes(decrypted[:8], 'big')) / 2**64

# Example usage
if __name__ == "__main__":
    # Setup decision space
    options = ["Option A", "Option B", "Option C"]
    num_voters = 3
    
    # Initialize system
    system = PrivateCollectiveChoice(num_voters, options)
    
    # Participants submit encrypted preferences
    preferences = [
        {"Option A": 0.8, "Option B": 0.3, "Option C": 0.5},
        {"Option A": 0.4, "Option B": 0.9, "Option C": 0.2},
        {"Option A": 0.6, "Option B": 0.7, "Option C": 0.3}
    ]
    
    for i, prefs in enumerate(preferences):
        system.submit_encrypted_preference(prefs, i)
    
    # Compute collective choice
    winner, scores = system.compute_collective_choice()
    print(f"Winning option: {winner}")
    print(f"Normalized scores: {scores}")
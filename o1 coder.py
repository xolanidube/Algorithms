import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple, Optional
import random

class TestCaseGenerator:
    """Test Case Generator (TCG) to create problem-oriented test samples"""
    def __init__(self, model_name: str = "deepseek-ai/deepseek-coder-1.3b-instruct"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def generate_test_cases(self, question: str, solution: str, num_cases: int = 3) -> List[Dict]:
        """Generate test cases for a given coding problem"""
        prompt = f"""
        ### Instruction
        Please generate {num_cases} test cases for the following coding problem.
        
        ### Problem
        {question}
        
        ### Solution
        {solution}
        
        ### Test Cases
        Generate diverse test cases that cover different scenarios.
        """
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,  # Changed from max_length
            num_return_sequences=num_cases,
            temperature=0.7,
            do_sample=True
        )
        
        test_cases = []
        for output in outputs:
            try:
                decoded = self.tokenizer.decode(output, skip_special_tokens=True)
                # More robust parsing with error handling
                parts = decoded.split("Input:")
                if len(parts) > 1:
                    input_output = parts[1].split("Output:")
                    if len(input_output) > 1:
                        test_cases.append({
                            "input": input_output[0].strip(),
                            "output": input_output[1].strip()
                        })
            except Exception as e:
                print(f"Error parsing test case: {e}")
                continue
            
        return test_cases

class ProcessRewardModel:
    """Process Reward Model (PRM) to evaluate quality of reasoning steps"""
    def __init__(self, model_name: str = "deepseek-ai/deepseek-coder-1.3b-instruct"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def evaluate_step(self, question: str, reasoning_steps: List[str], current_step: str) -> float:
        """Evaluate the quality of the current reasoning step"""
        context = "\n".join(reasoning_steps) if reasoning_steps else "No previous steps"
        prompt = f"""
        Rate the following reasoning step for solving a coding problem (respond with a single number between 0 and 1):
        
        Problem: {question}
        Previous steps: {context}
        Current step: {current_step}
        
        Rating (0-1):"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        try:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,  # Only need a short response for the rating
                num_return_sequences=1,
                temperature=0.1,  # Lower temperature for more consistent ratings
                do_sample=False  # Deterministic output for ratings
            )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract the last number from the generated text
            numbers = [float(s) for s in generated_text.split() if s.replace('.', '').isdigit()]
            if numbers:
                return min(max(numbers[-1], 0), 1)  # Ensure the score is between 0 and 1
            return 0.5  # Default score if no valid number is found
        except Exception as e:
            print(f"Error in evaluate_step: {e}")
            return 0.5  # Return default score on error

class PolicyModel:
    """Policy model that generates reasoning steps and code"""
    def __init__(self, model_name: str = "deepseek-ai/deepseek-coder-1.3b-instruct"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def generate_step(self, question: str, previous_steps: List[str]) -> str:
        """Generate next reasoning step"""
        context = "\n".join(previous_steps) if previous_steps else "No previous steps"
        prompt = f"""
        Generate the next logical reasoning step for solving this coding problem.
        
        Question: {question}
        Previous steps: {context}
        
        Next step:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        try:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Error in generate_step: {e}")
            return "Unable to generate next step"
    
    def generate_code(self, question: str, reasoning_steps: List[str]) -> str:
        """Generate final code based on reasoning steps"""
        context = "\n".join(reasoning_steps) if reasoning_steps else "No reasoning steps"
        prompt = f"""
        Generate Python code to solve this problem based on the reasoning steps.
        
        Problem: {question}
        Reasoning steps:
        {context}
        
        Python solution:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        try:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.2,  # Lower temperature for code generation
                do_sample=True
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Error in generate_code: {e}")
            return "# Error generating code"

# Rest of the classes (MCTS and O1Coder) remain the same
class MCTS:
    """Monte Carlo Tree Search for exploring reasoning paths"""
    def __init__(self, policy_model: PolicyModel, prm: ProcessRewardModel):
        self.policy_model = policy_model
        self.prm = prm
        
    def search(self, question: str, max_steps: int = 5, num_simulations: int = 10) -> List[str]:
        """Perform MCTS to find the best reasoning path"""
        best_path = []
        best_reward = float('-inf')
        
        for _ in range(num_simulations):
            current_path = []
            total_reward = 0
            
            for step in range(max_steps):
                # Generate next step using policy
                next_step = self.policy_model.generate_step(question, current_path)
                
                # Evaluate step quality
                reward = self.prm.evaluate_step(question, current_path, next_step)
                total_reward += reward
                
                current_path.append(next_step)
                
                # Early stopping if we've found a good path
                if reward < 0.2:  # Threshold for bad step
                    break
                    
            if total_reward > best_reward:
                best_reward = total_reward
                best_path = current_path.copy()
                
        return best_path

class O1Coder:
    """Main class implementing the O1-CODER framework"""
    def __init__(self):
        self.tcg = TestCaseGenerator()
        self.prm = ProcessRewardModel()
        self.policy_model = PolicyModel()
        self.mcts = MCTS(self.policy_model, self.prm)
        
    def solve(self, question: str) -> Tuple[List[str], str]:
        """Solve a coding problem using the framework"""
        try:
            # Find reasoning path using MCTS
            print("Generating reasoning path...")
            reasoning_path = self.mcts.search(question)
            
            # Generate code using the reasoning path
            print("Generating code...")
            code = self.policy_model.generate_code(question, reasoning_path)
            
            return reasoning_path, code
        except Exception as e:
            print(f"Error in solve: {e}")
            return ["Error generating solution"], "# Error occurred"

if __name__ == "__main__":
    # Initialize the framework
    print("Initializing O1-CODER...")
    o1_coder = O1Coder()
    
    # Example question
    question = """
    Write a function that takes two sorted arrays and merges them into
    a single sorted array without using any extra space.
    """
    
    # Solve the problem
    print("\nSolving problem...")
    reasoning_path, code = o1_coder.solve(question)
    
    print("\nReasoning Path:")
    for i, step in enumerate(reasoning_path, 1):
        print(f"\nStep {i}:")
        print(step)
    
    print("\nGenerated Code:")
    print(code)
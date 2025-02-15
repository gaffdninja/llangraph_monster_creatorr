import sys
import os
import io
from contextlib import contextmanager
from monster_agent import generate_amazing_monster, MonsterGenerator

@contextmanager
def simulate_user_input(inputs):
    """
    Context manager to simulate user inputs for testing.
    
    Args:
        inputs (list): A list of strings to simulate user inputs
    """
    # Backup the original stdin
    original_stdin = sys.stdin
    
    try:
        # Create a StringIO object with the predefined inputs
        input_stream = io.StringIO('\n'.join(inputs) + '\n')
        sys.stdin = input_stream
        
        yield
    finally:
        # Restore the original stdin
        sys.stdin = original_stdin

def test_monster_generation(narrative_inputs=None):
    """
    Test monster generation with optional predefined narrative inputs.
    
    Args:
        narrative_inputs (list, optional): List of 5 narrative input strings
    """
    if narrative_inputs is None:
        # Default narrative inputs if none are provided
        narrative_inputs = [
            "A forgotten ritual gone wrong",  # Dark secret
            "Misty mountain peaks shrouded in eternal fog",  # Environment
            "Collecting lost memories of ancient civilizations",  # Motivation
            "Observes and studies other creatures from a distance",  # Interaction
            "It can absorb and weaponize the deepest fears of its victims"  # Terrifying aspect
        ]
    
    # Ensure we have exactly 5 inputs
    assert len(narrative_inputs) == 5, "Must provide exactly 5 narrative inputs"
    
    # Use the context manager to simulate user inputs
    with simulate_user_input(narrative_inputs):
        monster = generate_amazing_monster()
    
    return monster

def batch_generate_monsters(num_monsters=2):
    """
    Generate multiple monsters with different narrative inputs.
    
    Args:
        num_monsters (int, optional): Number of monsters to generate. Defaults to 2.
    """
    generated_monsters = []
    
    # Predefined sets of narrative inputs
    input_variations = [
        [
            "A curse from an ancient witch",
            "Deep underwater caverns",
            "Seeking revenge on a forgotten enemy",
            "Manipulates other sea creatures through telepathy",
            "Can transform into the worst nightmare of its prey"
        ],
        [
            "A failed alchemical experiment",
            "Volcanic wastelands",
            "Collecting rare magical artifacts",
            "Communicates through complex pheromone signals",
            "Its very presence causes hallucinations and despair"
        ],
        [
            "A dimensional rift accident",
            "Floating islands in the sky",
            "Preserving the balance of cosmic energies",
            "Exists partially in multiple planes of reality",
            "Can rewrite the immediate reality around it"
        ],
        [
            "A betrayal by its own kind",
            "Endless desert with shifting sands",
            "Searching for a lost companion",
            "Mimics and learns from other intelligent beings",
            "Its form is constantly changing and unpredictable"
        ],
        [
            "A forbidden magical experiment",
            "Primordial forest with sentient plants",
            "Restoring an ancient ecosystem",
            "Communicates through root networks and spores",
            "Can control and merge with plant life"
        ]
    ]
    
    for inputs in input_variations[:num_monsters]:
        monster = test_monster_generation(inputs)
        if monster:
            generated_monsters.append(monster)
    
    return generated_monsters

def main():
    # Generate a single monster with default inputs
    print("Generating a single monster with default inputs...")
    monster = test_monster_generation()
    
    # Generate multiple monsters with varied inputs
    print("\nGenerating a batch of monsters...")
    batch_monsters = batch_generate_monsters()
    print(f"Generated {len(batch_monsters)} monsters successfully.")

if __name__ == "__main__":
    main()

import os
import re
import json
from typing import List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, ConfigDict
import random
from dotenv import load_dotenv
import time

# Load API key from .env file
load_dotenv()

# Ensure you have set your Groq API key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "your-api-key-here")

class Monster(BaseModel):
    """Represents a unique D&D monster with detailed attributes."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str = Field(description="Unique and evocative monster name")
    size: str = Field(description="Monster's size category")
    type: str = Field(description="Monster's creature type")
    alignment: str = Field(description="Monster's moral and ethical alignment")
    armor_class: int = Field(description="Monster's defensive capability")
    hit_points: int = Field(description="Monster's total health")
    speed: Dict[str, int] = Field(description="Monster's movement capabilities")
    abilities: Dict[str, int] = Field(description="Monster's core ability scores")
    special_abilities: List[Dict[str, str]] = Field(description="Unique monster traits")
    actions: List[Dict[str, str]] = Field(description="Monster's combat actions")
    lore: str = Field(description="Backstory and ecological context")

class MonsterGenerationState(BaseModel):
    """State for tracking monster generation process."""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra='allow'  # Allow extra fields for debugging
    )
    
    initial_concept: Optional[str] = None
    monster_draft: Optional[Dict[str, Any]] = None
    refined_monster: Optional[Dict[str, Any]] = None
    user_narrative_inputs: Optional[Dict[str, str]] = None  # New field for user inputs

    def __repr__(self):
        """Provide a detailed string representation for debugging."""
        return (
            f"MonsterGenerationState(\n"
            f"  initial_concept: {self.initial_concept}\n"
            f"  monster_draft: {bool(self.monster_draft)}\n"
            f"  refined_monster: {bool(self.refined_monster)}\n"
            f"  user_narrative_inputs: {bool(self.user_narrative_inputs)}\n"
            ")"
        )

class MonsterGenerator:
    def __init__(self, model_name="deepseek-r1-distill-llama-70b"):
        self.llm = ChatGroq(model=model_name, temperature=0.9)
        self.parser = PydanticOutputParser(pydantic_object=Monster)
        self.narrative_questions = [
            "What dark secret haunts this monster's past?",
            "In what unique environment does this monster thrive?",
            "What is the monster's most unexpected motivation?",
            "How does this monster interact with other creatures?",
            "What makes this monster truly terrifying?"
        ]

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from text, handling various formats."""
        # Remove markdown code block markers if present
        text = text.replace('```json', '').replace('```', '').strip()
        
        # Try to find JSON block between {}
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                # Remove any text before or after the JSON block
                json_str = json_match.group(0)
                return json.loads(json_str)
            except json.JSONDecodeError:
                print("JSON Extraction Failed. Problematic text:")
                print(text)
                raise

        # If direct JSON extraction fails, try to parse the entire text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Last resort: print the problematic text for debugging
            print("Could not parse JSON. Problematic text:")
            print(text)
            raise

    def get_user_narrative_inputs(self, state: MonsterGenerationState) -> MonsterGenerationState:
        """Collect user narrative inputs for monster generation."""
        print("\n🐉 Monster Creation Narrative Input 🐉")
        print("Please answer these 5 narrative questions to help shape your monster:\n")
        
        user_inputs = {}
        for i, question in enumerate(self.narrative_questions, 1):
            user_input = input(f"{i}. {question}\n   > ").strip()
            user_inputs[f"question_{i}"] = user_input
        
        new_state = MonsterGenerationState(
            initial_concept=state.initial_concept,
            monster_draft=state.monster_draft,
            refined_monster=state.refined_monster,
            user_narrative_inputs=user_inputs
        )
        
        return new_state

    def incorporate_narrative_inputs(self, state: MonsterGenerationState) -> MonsterGenerationState:
        """Incorporate user narrative inputs into the monster generation process."""
        if not state.user_narrative_inputs:
            return state
        
        narrative_prompt = ChatPromptTemplate.from_template(
            "Use these narrative inputs to enhance the monster's concept and lore:\n"
            "{narrative_inputs}\n\n"
            "Original Monster Concept: {initial_concept}\n\n"
            "Incorporate these narrative details into the monster's backstory, motivations, "
            "and unique characteristics. Provide a refined concept that integrates these inputs.\n\n"
            "Refined Concept: "
        )
        
        chain = narrative_prompt | self.llm
        refined_concept = chain.invoke({
            "narrative_inputs": "\n".join([f"Q: {q}\nA: {a}" for q, a in state.user_narrative_inputs.items()]),
            "initial_concept": state.initial_concept or "A mysterious and unique monster"
        }).content
        
        new_state = MonsterGenerationState(
            initial_concept=refined_concept,
            monster_draft=state.monster_draft,
            refined_monster=state.refined_monster,
            user_narrative_inputs=state.user_narrative_inputs
        )
        
        print("🌟 Narrative inputs incorporated into monster concept 🌟")
        return new_state

    def generate_concept(self, state: MonsterGenerationState) -> MonsterGenerationState:
        """Generate an initial monster concept."""
        concept_prompt = ChatPromptTemplate.from_template(
            "Create a unique and imaginative D&D monster concept. "
            "Provide a brief description that captures its essence. "
            "Make it something unexpected and exciting!\n\n"
            "Concept: "
        )
        
        chain = concept_prompt | self.llm
        concept = chain.invoke({}).content
        
        # Create a new state with the generated concept
        new_state = MonsterGenerationState(
            initial_concept=concept,
            monster_draft=state.monster_draft,
            refined_monster=state.refined_monster,
            user_narrative_inputs=state.user_narrative_inputs
        )
        
        print(f"Generated Concept: {concept}")
        return new_state

    def draft_monster(self, state: MonsterGenerationState) -> MonsterGenerationState:
        """Draft monster details based on the initial concept."""
        draft_prompt = ChatPromptTemplate.from_template(
            "Based on this monster concept: {concept}\n\n"
            "Draft a detailed monster using this JSON schema:\n{format_instructions}\n\n"
            "IMPORTANT INSTRUCTIONS:\n"
            "1. Provide ONLY a valid JSON object\n"
            "2. Do NOT include any explanatory text\n"
            "3. Ensure the JSON matches the exact schema provided\n"
            "4. Use realistic, balanced values for monster attributes"
        )
        
        chain = draft_prompt | self.llm
        monster_draft_text = chain.invoke({
            "concept": state.initial_concept,
            "format_instructions": self.parser.get_format_instructions()
        }).content
        
        # Extract JSON from the draft text
        try:
            monster_draft_dict = self._extract_json(monster_draft_text)
        except Exception as e:
            print(f"Draft Monster JSON Extraction Error: {e}")
            print(f"Problematic draft text: {monster_draft_text}")
            raise
        
        # Create a new state with the drafted monster
        new_state = MonsterGenerationState(
            initial_concept=state.initial_concept,
            monster_draft=monster_draft_dict,
            refined_monster=state.refined_monster,
            user_narrative_inputs=state.user_narrative_inputs
        )
        
        print("Monster Draft Created")
        return new_state

    def refine_monster(self, state: MonsterGenerationState) -> MonsterGenerationState:
        """Refine and balance the monster draft."""
        refine_prompt = ChatPromptTemplate.from_template(
            "Review and refine this monster draft:\n{draft}\n\n"
            "REFINEMENT INSTRUCTIONS:\n"
            "1. Carefully balance the monster's abilities and stats\n"
            "2. Ensure the monster is interesting and unique\n"
            "3. Provide ONLY a valid JSON object\n"
            "4. Do NOT include any additional text or explanations\n"
            "5. Maintain the exact JSON schema of the original draft"
        )
        
        chain = refine_prompt | self.llm
        
        refined_response = chain.invoke({
            "draft": json.dumps(state.monster_draft, indent=2)
        }).content
        
        # Extract JSON from the refined response
        try:
            refined_monster_dict = self._extract_json(refined_response)
        except Exception as e:
            print(f"Refined Monster JSON Extraction Error: {e}")
            print(f"Problematic refined text: {refined_response}")
            raise
        
        # Create a new state with the refined monster
        new_state = MonsterGenerationState(
            initial_concept=state.initial_concept,
            monster_draft=state.monster_draft,
            refined_monster=refined_monster_dict,
            user_narrative_inputs=state.user_narrative_inputs
        )
        
        print("Monster Refined Successfully")
        return new_state

def create_monster_generation_graph():
    """
    Create the LangGraph workflow for monster generation.
    
    The workflow now includes user narrative input collection and incorporation.
    """
    workflow = StateGraph(MonsterGenerationState)
    
    generator = MonsterGenerator()
    
    workflow.add_node("get_user_inputs", generator.get_user_narrative_inputs)
    workflow.add_node("generate_concept", generator.generate_concept)
    workflow.add_node("incorporate_narrative", generator.incorporate_narrative_inputs)
    workflow.add_node("draft_monster", generator.draft_monster)
    workflow.add_node("refine_monster", generator.refine_monster)
    
    workflow.set_entry_point("get_user_inputs")
    workflow.add_edge("get_user_inputs", "generate_concept")
    workflow.add_edge("generate_concept", "incorporate_narrative")
    workflow.add_edge("incorporate_narrative", "draft_monster")
    workflow.add_edge("draft_monster", "refine_monster")
    workflow.add_edge("refine_monster", END)
    
    return workflow.compile()

def generate_amazing_monster():
    """Generate and print an amazing D&D monster."""
    # Prompt for API key if not set
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        print("🔑 Groq API Key not found. Please visit https://console.groq.com to get your API key.")
        api_key = input("Enter your Groq API Key: ").strip()
        if not api_key:
            print("❌ No API key provided. Cannot generate monster.")
            return None
        os.environ["GROQ_API_KEY"] = api_key

    # Initialize the state
    initial_state = MonsterGenerationState(
        initial_concept="Create a unique and unexpected D&D monster",
        monster_draft=None,
        refined_monster=None,
        user_narrative_inputs=None
    )

    # Create the monster generation graph
    monster_graph = create_monster_generation_graph()

    try:
        # Run the monster generation workflow
        refined_monster = monster_graph.invoke(initial_state)
        
        # Ensure refined_monster is a dictionary
        if not isinstance(refined_monster, dict):
            print("❌ Monster generation did not produce a valid monster dictionary.")
            return None
        
        # Ensure the monster has a name for the filename
        monster_name = refined_monster.get('name', 'unnamed_monster')
        
        # Prepare filenames
        base_filename = f"generated_monster_{int(time.time())}"
        filename = f"generated_monsters/{monster_name.lower().replace(' ', '_')}.json"
        md_filename = f"generated_monsters/{monster_name.lower().replace(' ', '_')}.md"
        
        # Ensure the generated_monsters directory exists
        os.makedirs("generated_monsters", exist_ok=True)
        
        print("🐉 AMAZING D&D MONSTER GENERATED! 🐉")
        print(json.dumps(refined_monster, indent=2))
        
        # Save JSON
        with open(filename, 'w') as f:
            json.dump(refined_monster, f, indent=2)
        
        # Save Markdown
        with open(md_filename, 'w') as f:
            f.write(f"# {monster_name}\n\n")
            
            f.write("## Basic Information\n")
            f.write(f"- **Size:** {refined_monster.get('size', 'Unknown')}\n")
            f.write(f"- **Type:** {refined_monster.get('type', 'Unknown')}\n")
            f.write(f"- **Alignment:** {refined_monster.get('alignment', 'Unknown')}\n")
            f.write(f"- **Armor Class:** {refined_monster.get('armor_class', 'Unknown')}\n")
            f.write(f"- **Hit Points:** {refined_monster.get('hit_points', 'Unknown')}\n\n")
            
            f.write(f"## Abilities\n")
            abilities = refined_monster.get('abilities', {})
            f.write(f"- **Strength:** {abilities.get('Strength', 'Unknown')}\n")
            f.write(f"- **Dexterity:** {abilities.get('Dexterity', 'Unknown')}\n")
            f.write(f"- **Constitution:** {abilities.get('Constitution', 'Unknown')}\n")
            f.write(f"- **Intelligence:** {abilities.get('Intelligence', 'Unknown')}\n")
            f.write(f"- **Wisdom:** {abilities.get('Wisdom', 'Unknown')}\n")
            f.write(f"- **Charisma:** {abilities.get('Charisma', 'Unknown')}\n\n")
            
            f.write(f"## Speed\n")
            speed = refined_monster.get('speed', {})
            for movement_type, value in speed.items():
                f.write(f"- **{movement_type.capitalize()}:** {value} ft.\n")
            f.write("\n")
            
            f.write(f"## Special Abilities\n")
            special_abilities = refined_monster.get('special_abilities', [])
            for ability in special_abilities:
                # Handle both dictionary and string formats
                if isinstance(ability, dict):
                    name = ability.get('name', 'Unnamed Ability')
                    description = ability.get('description', 'No description available')
                elif isinstance(ability, str):
                    name = 'Unnamed Ability'
                    description = ability
                else:
                    continue
                
                f.write(f"### {name}\n")
                f.write(f"{description}\n\n")
            
            f.write(f"## Actions\n")
            actions = refined_monster.get('actions', [])
            for action in actions:
                # Handle both dictionary and string formats
                if isinstance(action, dict):
                    name = action.get('name', 'Unnamed Action')
                    description = action.get('description', 'No description available')
                elif isinstance(action, str):
                    name = 'Unnamed Action'
                    description = action
                else:
                    continue
                
                f.write(f"### {name}\n")
                f.write(f"{description}\n\n")
            
            f.write(f"## Lore\n")
            f.write(f"{refined_monster.get('lore', 'No lore available')}\n")
        
        print(f"🗄️ Monster details saved to {filename} and {md_filename}")
        
        return refined_monster
    
    except Exception as e:
        print(f"❌ Error during monster generation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    generate_amazing_monster()

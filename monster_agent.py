import os
import re
import json
import random
import time
from typing import List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv

# Load environment variables and API key
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "your-api-key-here")

###############################################################################
# Core Data Models and State Definitions
###############################################################################

class Monster(BaseModel):
    """
    Represents a unique D&D monster with detailed attributes.
    """
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
    """
    Tracks the state during monster generation.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra='allow'
    )
    
    initial_concept: Optional[str] = None
    monster_draft: Optional[Dict[str, Any]] = None
    refined_monster: Optional[Dict[str, Any]] = None
    user_narrative_inputs: Optional[Dict[str, str]] = None  # User inputs for narrative

    def __repr__(self):
        return (
            f"MonsterGenerationState(\n"
            f"  initial_concept: {self.initial_concept}\n"
            f"  monster_draft: {bool(self.monster_draft)}\n"
            f"  refined_monster: {bool(self.refined_monster)}\n"
            f"  user_narrative_inputs: {bool(self.user_narrative_inputs)}\n"
            ")"
        )

###############################################################################
# Monster Generator: Workflow Steps
###############################################################################

class MonsterGenerator:
    """
    Generates a D&D monster by directly using user narrative inputs in the concept.
    """
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
        text = text.strip()
        text = text.replace('```json', '').replace('```', '').strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                start = text.find('{')
                end = text.rfind('}') + 1
                if start != -1 and end != -1:
                    json_str = text[start:end]
                    json_str = json_str.replace('\n', ' ')
                    json_str = re.sub(r',\s*}', '}', json_str)
                    json_str = re.sub(r',\s*]', ']', json_str)
                    json_str = re.sub(r'(?<=")\s*(?=")|\s*(?<=\d)\s*(?=")|\s*(?<=])\s*(?=")', ',', json_str)
                    return json.loads(json_str)
            except Exception as e:
                print(f"Advanced JSON extraction failed: {e}")
        raise ValueError(f"Could not extract valid JSON from text: {text[:500]}...")

    def get_user_narrative_inputs(self, state: MonsterGenerationState) -> MonsterGenerationState:
        """Collect narrative inputs from the user."""
        print("\n🐉 Monster Creation Narrative Input 🐉")
        print("Please answer these 5 narrative questions to help shape your monster:\n")
        
        user_inputs = {}
        for i, question in enumerate(self.narrative_questions, 1):
            user_input = input(f"{i}. {question}\n   > ").strip()
            user_inputs[f"question_{i}"] = user_input
        
        return MonsterGenerationState(
            initial_concept=state.initial_concept,
            monster_draft=state.monster_draft,
            refined_monster=state.refined_monster,
            user_narrative_inputs=user_inputs
        )

    def generate_concept(self, state: MonsterGenerationState) -> MonsterGenerationState:
        """
        Generate an initial monster concept by directly incorporating 
        user narrative inputs into the prompt.
        """
        narrative_context = ""
        if state.user_narrative_inputs:
            narrative_context = "\n".join([
                f"Dark Secret: {state.user_narrative_inputs.get('question_1', '')}",
                f"Unique Environment: {state.user_narrative_inputs.get('question_2', '')}",
                f"Unexpected Motivation: {state.user_narrative_inputs.get('question_3', '')}",
                f"Creature Interaction: {state.user_narrative_inputs.get('question_4', '')}",
                f"Terrifying Aspect: {state.user_narrative_inputs.get('question_5', '')}"
            ]) + "\n\n"
        
        # Directly include narrative context in the concept prompt
        concept_prompt = ChatPromptTemplate.from_template(
            f"{narrative_context}"
            "Create a unique and imaginative D&D monster concept that fully reflects the narrative inputs above. "
            "Provide a brief description that captures its essence, ensuring the concept is aligned with the user's creative ideas.\n\n"
            "Concept: "
        )
        
        chain = concept_prompt | self.llm
        concept = chain.invoke({}).content
        
        print(f"Generated Concept: {concept}")
        return MonsterGenerationState(
            initial_concept=concept,
            monster_draft=state.monster_draft,
            refined_monster=state.refined_monster,
            user_narrative_inputs=state.user_narrative_inputs
        )

    def draft_monster(self, state: MonsterGenerationState) -> MonsterGenerationState:
        """
        Draft detailed monster information based on the generated concept.
        """
        narrative_context = ""
        if state.user_narrative_inputs:
            narrative_context = (
                "User Provided Narrative Context:\n"
                f"1. Dark Secret: {state.user_narrative_inputs.get('question_1', 'No dark secret provided')}\n"
                f"2. Unique Environment: {state.user_narrative_inputs.get('question_2', 'No unique environment specified')}\n"
                f"3. Unexpected Motivation: {state.user_narrative_inputs.get('question_3', 'No unexpected motivation given')}\n"
                f"4. Creature Interaction: {state.user_narrative_inputs.get('question_4', 'No interaction details provided')}\n"
                f"5. Terrifying Aspect: {state.user_narrative_inputs.get('question_5', 'No terrifying aspect described')}\n\n"
            )
        full_concept = f"{narrative_context}Monster Concept: {state.initial_concept}"
        
        draft_prompt = ChatPromptTemplate.from_template(
            "Based on this detailed monster concept and user narrative inputs:\n{full_concept}\n\n"
            "GENERATE A COMPLETE AND DETAILED MONSTER WITH THE FOLLOWING REQUIREMENTS:\n"
            "1. Create a UNIQUE and EVOCATIVE monster name that reflects its nature\n"
            "2. Provide a comprehensive monster description\n"
            "3. Use the JSON schema:\n{format_instructions}\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "- NAME MUST be unique, descriptive, and memorable\n"
            "- Integrate user's narrative inputs into EVERY aspect of the monster\n"
            "- Ensure ALL fields are populated with meaningful, creative content\n"
            "- NO 'Unknown' or placeholder values allowed\n"
            "- Monster should be fully realized and ready for D&D gameplay"
        )
        
        chain = draft_prompt | self.llm
        monster_draft_text = chain.invoke({
            "full_concept": full_concept,
            "format_instructions": self.parser.get_format_instructions()
        }).content
        
        try:
            monster_draft_dict = self._extract_json(monster_draft_text)
            # Ensure a unique and meaningful monster name is provided
            if not monster_draft_dict.get('name') or monster_draft_dict['name'].lower().startswith('unnamed'):
                name_parts = []
                if monster_draft_dict.get('type'):
                    name_parts.append(monster_draft_dict['type'].capitalize())
                if monster_draft_dict.get('size'):
                    name_parts.append(monster_draft_dict['size'])
                unique_suffix = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=3))
                monster_draft_dict['name'] = f"The {' '.join(name_parts)} of {unique_suffix}"
            
            critical_fields = [
                'size', 'type', 'alignment', 'armor_class', 'hit_points', 
                'speed', 'abilities', 'special_abilities', 'actions', 'lore'
            ]
            for field in critical_fields:
                if not monster_draft_dict.get(field):
                    default_values = {
                        'size': 'Medium',
                        'type': 'Monstrosity',
                        'alignment': 'Unaligned',
                        'armor_class': 12,
                        'hit_points': 45,
                        'speed': {'walk': 30},
                        'abilities': {
                            'str': 10, 'dex': 10, 'con': 10,
                            'int': 10, 'wis': 10, 'cha': 10
                        },
                        'special_abilities': [{
                            'name': 'Adaptive Nature',
                            'description': 'The creature can adapt to its environment, gaining slight advantages.'
                        }],
                        'actions': [{
                            'name': 'Basic Attack',
                            'description': 'Melee or ranged attack with standard damage.'
                        }],
                        'lore': 'A mysterious creature with an intriguing backstory waiting to be discovered.'
                    }
                    monster_draft_dict[field] = default_values.get(field, f"Default {field}")
        except Exception as e:
            print(f"Draft Monster JSON Extraction Error: {e}")
            print(f"Problematic draft text: {monster_draft_text}")
            raise
        
        print(f"🐉 Monster draft created: {monster_draft_dict.get('name', 'Unnamed Monster')} 🐉")
        return MonsterGenerationState(
            initial_concept=state.initial_concept,
            monster_draft=monster_draft_dict,
            refined_monster=state.refined_monster,
            user_narrative_inputs=state.user_narrative_inputs
        )

    def refine_monster(self, state: MonsterGenerationState) -> MonsterGenerationState:
        """
        Refine and balance the monster draft to produce the final monster.
        """
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
        
        try:
            refined_monster_dict = self._extract_json(refined_response)
        except Exception as e:
            print(f"Refined Monster JSON Extraction Error: {e}")
            print(f"Problematic refined text: {refined_response}")
            raise
        
        print("Monster Refined Successfully")
        return MonsterGenerationState(
            initial_concept=state.initial_concept,
            monster_draft=state.monster_draft,
            refined_monster=refined_monster_dict,
            user_narrative_inputs=state.user_narrative_inputs
        )


###############################################################################
# Utility Function for Concatenating Markdown Content
###############################################################################

def concat_markdown_content(markdown_content: List[Dict[str, str]]) -> str:
    """
    Concatenates multiple markdown content entries into one formatted string.
    """
    final_content = ""
    separator = "=" * 40
    for content_dict in markdown_content:
        url = content_dict["url"]
        markdown_text = content_dict["markdown_text"]
        title = content_dict["title"]
        final_content += (
            f"{separator}\n URL: {url}\n Page Title: {title}\n Markdown:\n"
            f"{separator}\n{markdown_text}\n"
        )
    return final_content

###############################################################################
# Graph Construction: Monster Generation Workflow
###############################################################################

def create_monster_generation_graph():
    """
    Create the LangGraph workflow for monster generation.
    
    The workflow now includes:
      - User narrative input collection
      - Concept generation (which now directly uses user inputs)
      - Monster drafting
      - Monster refinement
      
    This follows a graph-based structure with defined nodes and edges.
    """
    workflow = StateGraph(MonsterGenerationState)
    generator = MonsterGenerator()
    
    workflow.add_node("get_user_inputs", generator.get_user_narrative_inputs)
    workflow.add_node("generate_concept", generator.generate_concept)
    workflow.add_node("draft_monster", generator.draft_monster)
    workflow.add_node("refine_monster", generator.refine_monster)
    
    # Workflow: user inputs -> generate concept -> draft monster -> refine monster
    workflow.set_entry_point("get_user_inputs")
    workflow.add_edge("get_user_inputs", "generate_concept")
    workflow.add_edge("generate_concept", "draft_monster")
    workflow.add_edge("draft_monster", "refine_monster")
    workflow.add_edge("refine_monster", END)
    
    return workflow.compile()

###############################################################################
# Main Function: Generate and Save the Monster
###############################################################################

def generate_amazing_monster():
    """
    Generate and print an amazing D&D monster.
    
    This function integrates LangChain's LLM capabilities with LangGraph's orchestration,
    ensuring that user narrative inputs directly influence the monster concept.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        print("🔑 Groq API Key not found. Please visit https://console.groq.com to get your API key.")
        api_key = input("Enter your Groq API Key: ").strip()
        if not api_key:
            print("❌ No API key provided. Cannot generate monster.")
            return None
        os.environ["GROQ_API_KEY"] = api_key

    initial_state = MonsterGenerationState(
        initial_concept="Create a unique and unexpected D&D monster",
        monster_draft=None,
        refined_monster=None,
        user_narrative_inputs=None
    )

    # Build the monster generation workflow graph
    monster_graph = create_monster_generation_graph()

    try:
        final_state = monster_graph.invoke(initial_state)
        # Use dictionary-style access to retrieve the refined monster
        refined_monster = final_state["refined_monster"]
        if not isinstance(refined_monster, dict):
            print("❌ Monster generation did not produce a valid monster dictionary.")
            return None

        monster_name = refined_monster.get('name', f'unnamed_monster_{int(time.time())}')
        base_filename = monster_name.lower().replace(' ', '_')
        filename = f"generated_monsters/{base_filename}.json"
        md_filename = f"generated_monsters/{base_filename}.md"
        
        os.makedirs("generated_monsters", exist_ok=True)
        
        print("🐉 AMAZING D&D MONSTER GENERATED! 🐉")
        print(json.dumps(refined_monster, indent=2))
        
        # Save JSON output
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(refined_monster, f, indent=2, ensure_ascii=False)
        
        # Save Markdown output with improved formatting to avoid "Unnamed" labels
        with open(md_filename, 'w', encoding='utf-8') as f:
            f.write(f"# {monster_name}\n\n")
            f.write("## Basic Information\n\n")
            basic_info = [
                f"- **Size:** {refined_monster.get('size', 'Unknown')}\n",
                f"- **Type:** {refined_monster.get('type', 'Unknown')}\n",
                f"- **Alignment:** {refined_monster.get('alignment', 'Unknown')}\n",
                f"- **Armor Class:** {refined_monster.get('armor_class', 'Unknown')}\n",
                f"- **Hit Points:** {refined_monster.get('hit_points', 'Unknown')}\n"
            ]
            f.writelines(basic_info)
            f.write("\n")
            
            f.write("## Abilities\n\n")
            abilities = refined_monster.get('abilities', {})
            # Map abbreviated keys to full names
            ability_keys = {
                'Strength': 'str',
                'Dexterity': 'dex',
                'Constitution': 'con',
                'Intelligence': 'int',
                'Wisdom': 'wis',
                'Charisma': 'cha'
            }
            for full_name, abbr in ability_keys.items():
                value = abilities.get(abbr, abilities.get(full_name, 'Unknown'))
                f.write(f"- **{full_name}:** {value}\n")
            f.write("\n")
            
            f.write("## Speed\n\n")
            speed = refined_monster.get('speed', {})
            for movement_type, value in speed.items():
                f.write(f"- **{movement_type.capitalize()}:** {value} ft.\n")
            f.write("\n")
            
            f.write("## Special Abilities\n\n")
            special_abilities = refined_monster.get('special_abilities', [])
            if special_abilities:
                for ability in special_abilities:
                    if isinstance(ability, dict):
                        name = ability.get('name', '').strip()
                        description = ability.get('description', '').strip()
                        if not name and ":" in description:
                            parts = description.split(":", 1)
                            name = parts[0].strip()
                            description = parts[1].strip()
                        elif not name:
                            name = "Special Ability"
                        f.write(f"### {name}\n{description}\n\n")
                    elif isinstance(ability, str):
                        if ":" in ability:
                            parts = ability.split(":", 1)
                            name = parts[0].strip()
                            description = parts[1].strip()
                        else:
                            name = "Special Ability"
                            description = ability.strip()
                        f.write(f"### {name}\n{description}\n\n")
            else:
                f.write("*No special abilities found.*\n\n")
            
            f.write("## Actions\n\n")
            actions = refined_monster.get('actions', [])
            if actions:
                for action in actions:
                    if isinstance(action, dict):
                        name = action.get('name', '').strip()
                        description = action.get('description', '').strip()
                        if not name and ":" in description:
                            parts = description.split(":", 1)
                            name = parts[0].strip()
                            description = parts[1].strip()
                        elif not name:
                            name = "Action"
                        f.write(f"### {name}\n{description}\n\n")
                    elif isinstance(action, str):
                        if ":" in action:
                            parts = action.split(":", 1)
                            name = parts[0].strip()
                            description = parts[1].strip()
                        else:
                            name = "Action"
                            description = action.strip()
                        f.write(f"### {name}\n{description}\n\n")
            else:
                f.write("*No actions found.*\n\n")
            
            f.write("## Lore\n\n")
            lore = refined_monster.get('lore', 'No lore available')
            f.write(f"{lore}\n")
        
        print(f"🗄️ Monster saved as {filename} and {md_filename}")
        return refined_monster
    
    except Exception as e:
        print(f"❌ Monster generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

###############################################################################
# Entry Point
###############################################################################

if __name__ == "__main__":
    generate_amazing_monster()

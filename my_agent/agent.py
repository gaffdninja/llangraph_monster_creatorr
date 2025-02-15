import os
import json
import time
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from my_agent.monster import MonsterGenerationState, MonsterGenerator
from my_agent.utils import concat_markdown_content
from my_agent.config import LANGCHAIN_MODEL, DEFAULT_INITIAL_CONCEPT

# Load environment variables from .env file
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "your-api-key-here")

def create_monster_generation_graph():
    """
    Create the LangGraph workflow for monster generation.
    The workflow includes:
      - User narrative input collection
      - Concept generation (directly using user inputs)
      - Monster drafting
      - Monster refinement
    """
    generator = MonsterGenerator(model_name=LANGCHAIN_MODEL)
    workflow = StateGraph(MonsterGenerationState)
    
    workflow.add_node("get_user_inputs", generator.get_user_narrative_inputs)
    workflow.add_node("generate_concept", generator.generate_concept)
    workflow.add_node("draft_monster", generator.draft_monster)
    workflow.add_node("refine_monster", generator.refine_monster)
    
    # Build workflow: user inputs -> generate concept -> draft monster -> refine monster
    workflow.set_entry_point("get_user_inputs")
    workflow.add_edge("get_user_inputs", "generate_concept")
    workflow.add_edge("generate_concept", "draft_monster")
    workflow.add_edge("draft_monster", "refine_monster")
    workflow.add_edge("refine_monster", END)
    
    return workflow.compile()

def generate_amazing_monster():
    """
    Generate and print an amazing D&D monster.
    This function integrates LangChain's LLM capabilities with LangGraph's orchestration,
    ensuring that user narrative inputs directly influence the monster concept.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        print("GROQ API Key not found. Please set your key in .env.")
        api_key = input("Enter your Groq API Key: ").strip()
        if not api_key:
            print("No API key provided. Exiting.")
            return None
        os.environ["GROQ_API_KEY"] = api_key

    initial_state = MonsterGenerationState(
        initial_concept=DEFAULT_INITIAL_CONCEPT,
        monster_draft=None,
        refined_monster=None,
        user_narrative_inputs=None
    )

    graph = create_monster_generation_graph()

    try:
        final_state = graph.invoke(initial_state)
        # Retrieve the refined monster from the final state (using dictionary access)
        refined_monster = final_state["refined_monster"]
        if not isinstance(refined_monster, dict):
            print("Monster generation did not produce a valid monster dictionary.")
            return None

        monster_name = refined_monster.get("name", f"unnamed_monster_{int(time.time())}")
        base_filename = monster_name.lower().replace(" ", "_")
        json_filename = f"generated_monsters/{base_filename}.json"
        md_filename = f"generated_monsters/{base_filename}.md"
        
        os.makedirs("generated_monsters", exist_ok=True)
        
        print("🐉 AMAZING D&D MONSTER GENERATED! 🐉")
        print(json.dumps(refined_monster, indent=2))
        
        # Save JSON output
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(refined_monster, f, indent=2, ensure_ascii=False)
        
        # Generate Markdown output with improved formatting to avoid "Unnamed" labels
        with open(md_filename, "w", encoding="utf-8") as f:
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
            abilities = refined_monster.get("abilities", {})
            ability_keys = {
                "Strength": "str",
                "Dexterity": "dex",
                "Constitution": "con",
                "Intelligence": "int",
                "Wisdom": "wis",
                "Charisma": "cha"
            }
            for full_name, abbr in ability_keys.items():
                value = abilities.get(abbr, abilities.get(full_name, "Unknown"))
                f.write(f"- **{full_name}:** {value}\n")
            f.write("\n")
            f.write("## Speed\n\n")
            speed = refined_monster.get("speed", {})
            for movement_type, value in speed.items():
                f.write(f"- **{movement_type.capitalize()}:** {value} ft.\n")
            f.write("\n")
            f.write("## Special Abilities\n\n")
            special_abilities = refined_monster.get("special_abilities", [])
            if special_abilities:
                for ability in special_abilities:
                    if isinstance(ability, dict):
                        name = ability.get("name", "").strip()
                        description = ability.get("description", "").strip()
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
            actions = refined_monster.get("actions", [])
            if actions:
                for action in actions:
                    if isinstance(action, dict):
                        name = action.get("name", "").strip()
                        description = action.get("description", "").strip()
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
            lore = refined_monster.get("lore", "No lore available")
            f.write(f"{lore}\n")
        
        print(f"🗄️ Monster saved as {json_filename} and {md_filename}")
        return refined_monster
    except Exception as e:
        print(f"Monster generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    generate_amazing_monster()

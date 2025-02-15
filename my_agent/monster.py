import os
import re
import json
import random
from typing import List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, ConfigDict

class Monster(BaseModel):
    """
    Represents a unique D&D monster with detailed attributes.
    This model is used to structure the LLM's output.
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
    Contains the initial concept, drafted monster, refined monster, and user narrative inputs.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='allow')
    
    initial_concept: Optional[str] = None
    monster_draft: Optional[Dict[str, Any]] = None
    refined_monster: Optional[Dict[str, Any]] = None
    user_narrative_inputs: Optional[Dict[str, str]] = None

    def __repr__(self):
        return (
            f"MonsterGenerationState(\n"
            f"  initial_concept: {self.initial_concept}\n"
            f"  monster_draft: {bool(self.monster_draft)}\n"
            f"  refined_monster: {bool(self.refined_monster)}\n"
            f"  user_narrative_inputs: {bool(self.user_narrative_inputs)}\n"
            ")"
        )

class MonsterGenerator:
    """
    Generates a D&D monster using a multi-step LangGraph workflow.
    Directly incorporates user narrative inputs into the concept generation.
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
        """
        Extract JSON from text, handling common formatting issues.
        """
        text = text.strip().replace('```json', '').replace('```', '').strip()
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
        """
        Collect narrative inputs from the user.
        """
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
        Generate an initial monster concept, directly incorporating user narrative inputs.
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

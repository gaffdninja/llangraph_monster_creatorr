# D&D Monster Generator 🐉✨

## Overview

This project is an AI-powered Dungeons & Dragons monster generator that uses advanced language models to create unique, detailed, and imaginative monsters for tabletop roleplaying games.

## Features

- 🤖 AI-driven monster generation
- 🎲 Comprehensive monster attribute creation
- 📊 Detailed monster statistics and lore
- 🌟 Unique monster concepts
- **NEW: Interactive Narrative Input**
  - 5 narrative questions to personalize monster generation
  - User-driven storytelling elements
  - Enhanced monster backstory and motivation

### Narrative Input Questions

When you run the monster generator, you'll be prompted to answer 5 key narrative questions:

1. What dark secret haunts this monster's past?
2. In what unique environment does this monster thrive?
3. What is the monster's most unexpected motivation?
4. How does this monster interact with other creatures?
5. What makes this monster truly terrifying?

Your answers will be integrated into the monster's concept, making each generated monster truly unique!

## Prerequisites

- Python 3.8+
- Groq API Key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/gaffdninja/llangraph_monster_creatorr.git
cd llangraph_monster_creatorr
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your environment variables:
- Create a `.env` file in the project root
- Add your Groq API key:
```
GROQ_API_KEY=your_groq_api_key_here
```

## Usage

Run the monster generator:
```bash
python monster_agent.py
```

## Testing and Automation

### Test Monster Generator

The `test_monster_generator.py` script provides several ways to automate monster generation:

#### Features
- Simulate user inputs for narrative questions
- Generate a single monster with default or custom inputs
- Batch generate multiple monsters with varied narrative inputs

#### Usage Examples

```bash
# Generate a single monster with default inputs
python test_monster_generator.py

# In your Python script
from test_monster_generator import test_monster_generation, batch_generate_monsters

# Generate a single monster with custom inputs
custom_inputs = [
    "A dark ritual gone wrong",
    "Misty mountain peaks",
    "Collecting lost memories",
    "Observes creatures from a distance",
    "Absorbs and weaponizes deepest fears"
]
monster = test_monster_generation(custom_inputs)

# Generate a batch of monsters
batch_monsters = batch_generate_monsters(num_monsters=3)
```

### Test Scenarios
- Default narrative inputs
- Custom narrative inputs
- Batch monster generation
- Error handling and input validation

## Generated Monsters

The project automatically generates unique monsters, storing them in the `generated_monsters/` directory:
- Chronochroma
- EchoFlora
- GlintWraith
- The Echokeeper

## Technologies

- LangGraph
- LangChain
- Groq AI
- Pydantic

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingMonster`)
3. Commit your changes (`git commit -m 'Add some amazing monster generation'`)
4. Push to the branch (`git push origin feature/AmazingMonster`)
5. Open a Pull Request

## License

[Specify your license here]

## Acknowledgments

- Inspired by the magical world of Dungeons & Dragons
- Powered by cutting-edge AI technology
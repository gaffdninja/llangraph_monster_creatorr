# D&D Monster Generator 🐉✨

## Overview

This project is an AI-powered Dungeons & Dragons monster generator that uses advanced language models to create unique, detailed, and imaginative monsters for tabletop roleplaying games.

## Features

- 🤖 AI-driven monster generation
- 🎲 Comprehensive monster attribute creation
- 📊 Detailed monster statistics and lore
- 🌟 Unique monster concepts

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
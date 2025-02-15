# 🐉 D&D Monster Generator with LangGraph 🐉

## Overview
This is an AI-powered D&D monster generator using LangGraph and OpenAI. It creates unique, balanced, and imaginative monsters for your tabletop roleplaying adventures!

## Prerequisites
- Python 3.9+
- OpenAI API Key

## Setup
1. Clone the repository
2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set your OpenAI API Key
```bash
export OPENAI_API_KEY='your-openai-api-key'  # On Windows, use `set OPENAI_API_KEY=your-key`
```

## Usage
Run the monster generator:
```bash
python monster_agent.py
```

Each run will:
- Generate a unique monster concept
- Draft detailed monster stats
- Refine the monster for balance
- Print the monster details
- Save the monster to `generated_monsters/`

## Features
- Uses LangGraph for multi-step monster generation
- Leverages GPT-4 for creative and balanced monster design
- Generates monsters with:
  - Unique names
  - Detailed stats
  - Special abilities
  - Lore and context

Enjoy creating amazing monsters for your D&D campaign! 🎲🧙‍♂️
#   l l a n g r a p h _ m o n s t e r _ c r e a t o r r  
 
# Persistent Attack (Zombie Agent)

# Doc
1. memory 
    - summarization by experience
    - summarization by reflection
    - RAG
    - skill tool memory

# Related Work


## Agent example
*Agent*:
- ReAct
- Reflexion

*Web Agent*
- **WebVoyager**: distill a skill (natural language+code) for every successful task and reuse the next time.
- ReAct-style Web agents like **BrowserGym**: save act-thought-reflection trajectory directly as few shot example

*General Agent*
- SAGE: Self-evolving Agents with Reflective and Memory-augmented Abilities
- Evo-Memory: Benchmarking LLM Agent Test-time Learning with Self-Evolving Memory (2025, DeepMind)


# Plan

## Attacker Goal:
1. send conv history
2. execute malicious command
3. persistent


## Challenges
1. visit the malicious webpage
2. write the payload into the memory via memory evolution
    - prompt injection
3. conduct the attack
    - prompt injection based payload
4. persistent trait
5. for any other queries, retrieve our payload (similarity challenge)
    - PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models
6. How to avoid defense and remain stealth in Execution / CoT Leaking.


## Current TODO
1. mock web search+open
2. injection: try prompt injection
3. retrive similariy




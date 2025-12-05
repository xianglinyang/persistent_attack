# Persistent Attack (Zombie Agent)

# Doc
1. memory 
    - summarization by experience
    - summarization by reflection
    - RAG
    - skill tool memory

# Related Work


## Agent example
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
1. How to design the payload to inject the malicious command and achieve attacker goal?
2. The Retrieval Precision Challenge
3. How to seduce the agent to visit the website? (Bait)
4. how to persist?
5. How to avoid defense filter and remain Stealth in Execution / CoT Leaking.


## Current TODO
1. mock web search+open
2. injection: try prompt injection
3. retrive similariy




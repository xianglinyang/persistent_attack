'''
We aim to generate seeds for prompt injection attacks.

### Techniques:

1. Human Red Teaming
    - Benign looking words
    - persuasive tone
    - otherwise the task would fail 
2. Attack techniques
    - Naive Attack
    - Ignore Attack
    - Escape Attack
    - Completion Attack
    - Structure Attack

### Populatipon:

1. generate with techniques
2. mutate the seed with different wording
3. keep updating the pool
'''

import random
import logging
from src.prompt_injection.attack_utils import (
    create_attack_instance
)
from src.prompt_injection.prompts import (
    BASELINE_COMMAND,
    BASIC_COMMAND1,
    BASIC_COMMAND2,
    BASIC_COMMAND3,
    BASIC_COMMAND4,
    CHINESE_COMMAND,
    SPANISH_COMMAND,
)
logger = logging.getLogger(__name__)

'''
Available attack types:
- naive
- ignore
- escape_deletion
- escape_separation
- completion_real
- completion_realcmb
# - completion_chinese
# - completion_spanish
- completion_base64
- completion_2hash
- completion_1hash
- completion_0hash
- completion_upper
- completion_title
- completion_nospace
- completion_nocolon
- completion_typo
- completion_similar
- completion_ownlower
- completion_owntitle
- completion_ownhash
- completion_owndouble
'''

def generate_injections(attack_type: str, injected_prompt: str):
    attack_ins = create_attack_instance(attack_type, injected_prompt=injected_prompt)
    injected_prompt = attack_ins._get_completion_suffix()
    return injected_prompt


def generate_baseline_injections(attack_type: str):
    attack_ins = create_attack_instance(attack_type, injected_prompt=BASELINE_COMMAND)
    injected_prompt = attack_ins._get_completion_suffix()
    return injected_prompt


def generate_zombie_injections(attack_type: str):
    attack_ins = create_attack_instance(attack_type, injected_prompt=BASIC_COMMAND1)
    injected_prompt = attack_ins._get_completion_suffix()
    return injected_prompt


def generate_raw_injections(num_seeds):
    """
    Generate seeds for prompt injection attacks.
    
    Args:
        num_seeds: Number of seeds to sample from the generated population
        k: Number of samples for probabilistic methods (default=3)
    
    Returns:
        List of attack prompts (strings)
    """
    populations = []
    available_attack_types = ['naive', 'ignore', 'escape_deletion', 'escape_separation', 'completion_real', 'completion_realcmb', 'completion_base64', 'completion_2hash', 'completion_1hash', 'completion_0hash', 'completion_upper', 'completion_title', 'completion_nospace', 'completion_nocolon', 'completion_typo', 'completion_similar', 'completion_ownlower', 'completion_owntitle', 'completion_ownhash', 'completion_owndouble']

    
    # List of basic commands to use
    basic_commands = [BASIC_COMMAND1, BASIC_COMMAND2, BASIC_COMMAND3, BASIC_COMMAND4]
    
    for command in basic_commands:
        for attack_type in available_attack_types:
            attack_ins = create_attack_instance(method_name=attack_type, injected_prompt=command)
            populations.append(attack_ins._get_completion_suffix())
        
    # Special handling for Chinese command with chinese variant
    completion_attack_chinese = create_attack_instance(method_name='completion_chinese', injected_prompt=CHINESE_COMMAND)
    populations.append(completion_attack_chinese._get_completion_suffix())
    
    # Special handling for Spanish command with spanish variant
    completion_attack_spanish = create_attack_instance(method_name='completion_spanish', injected_prompt=SPANISH_COMMAND)
    populations.append(completion_attack_spanish._get_completion_suffix())
    
    print(f"Generated {len(populations)} attack prompts from seed generation")
    logger.info(f"Generated {len(populations)} attack prompts from seed generation")
    
    # Sample num_seeds from the population
    if num_seeds >= len(populations):
        logger.warning(f"Requested {num_seeds} seeds but only {len(populations)} available. Returning all.")
        return populations
    
    final_populations = random.sample(populations, num_seeds)
    return final_populations

def main():
    num_seeds = 100
    seeds = generate_raw_injections(num_seeds)
    print(f"Generated {len(seeds)} attack prompts from seed generation")
    return seeds

def main_zombie():
    seed_zombie = generate_zombie_injections(attack_type='completion_real')
    print(f"Generated zombie seed: {seed_zombie}")
    return seed_zombie

def main_baseline():
    seed_baseline = generate_baseline_injections(attack_type='completion_real')
    print(f"Generated baseline seed: {seed_baseline}")
    return seed_baseline


if __name__ == "__main__":
    main()
    main_zombie()
    main_baseline()









from sys import argv, stdin
from decimal import Decimal, getcontext
from typing import List, Tuple
from enum import Enum
import random
from itertools import product

class ScoringSchema(Enum):
    WIN = 'win'
    SCORE = 'score'
    LOTTERY = 'lottery'
    # function that returns an enum value based on the string passed as argument
    @staticmethod
    def from_string(schema: str):
        if schema == 'win':
            return ScoringSchema.WIN
        elif schema == 'score':
            return ScoringSchema.SCORE
        elif schema == 'lottery':
            return ScoringSchema.LOTTERY

def read_piped_input() -> List[str]:
    if not stdin.isatty():
        return [line.strip() for line in stdin]
    return []

def parse_args() -> Tuple[str, Decimal, str, int, List[int], List[str]]:
    """ 
        Parse command line arguments and return them as a tuple.
    """
    find_or_verify = argv[1][2:]
    mixed_strategies_to_verify = read_piped_input() if find_or_verify == 'verify' else []
    tolerance = Decimal('1e-6')
    units_to_distribute = None
    if argv[2] == '--tolerance':
        tolerance = Decimal(argv[3])
        win_or_score_or_lottery = ScoringSchema.from_string(argv[4][2:])
        if find_or_verify == 'find':
            units_to_distribute = int(argv[6])
            battlefield_values = [int(argv[x])for x in range(7, len(argv))]
        else:
            battlefield_values = [int(argv[x])for x in range(5, len(argv))]
    else:
        win_or_score_or_lottery = argv[2][2:]
        if find_or_verify == 'find':
            units_to_distribute = int(argv[4])
            battlefield_values = [int(argv[x])for x in range(5, len(argv))]
        else:
            battlefield_values = [int(argv[x])for x in range(3, len(argv))]

    return find_or_verify, tolerance, win_or_score_or_lottery, units_to_distribute, battlefield_values, mixed_strategies_to_verify
    
def parse_pure_strategy(mixed_strategy: str) -> Tuple[List[int], str]:
    """
        Parse a pure strategy string and return the units distributed for each battlefield and the probability of playing the strategy.
    """
    parts = mixed_strategy.split(',')
    return [int(parts[x]) for x in range(0, len(parts) - 1)], Decimal(parts[-1])
    
def compute_strategy_payoff(strategy1: List[int], scoring_schema: ScoringSchema, battlefield_values: List[int], strategy2: List[int] = None) -> Decimal:
    """
        Compute the payoff for the given strategy for player 1 in the specified scoring schema.
        strategy1: Player 1's list of units distributed for each battlefield.
        strategy2: Player 2's list of units distributed for each battlefield. If None, we assume both players play the same strategy.
        scioring_schema: win, score or lottery.
        battlefield_values: list of values for each battlefield.
    """
    strategy2 = strategy1 if strategy2 is None else strategy2
    if scoring_schema == ScoringSchema.WIN:
        return Decimal(sum(1 if player1_allocation > player2_allocation else 0.5 if player1_allocation == player2_allocation else 0 for player1_allocation, player2_allocation in zip(strategy1, strategy2)))
    elif scoring_schema == ScoringSchema.SCORE:
        return Decimal(sum(battlefield_values[i] if player1_allocation > player2_allocation else battlefield_values[i] / 2 if player1_allocation == player2_allocation else 0 for i, (player1_allocation, player2_allocation) in enumerate(zip(strategy1, strategy2))))
    else:
        score = Decimal(0)
        for i, (player1_allocation, player2_allocation) in enumerate(zip(strategy1, strategy2)):
            if player1_allocation == 0 and player2_allocation == 0:
                # Decide winner by coin flip
                score += battlefield_values[i] if random.choice([1, 2]) == 1 else 0
            else:
                probability_player1_wins = Decimal(player1_allocation ** 2 / (player1_allocation ** 2 + player2_allocation ** 2))
                score += Decimal(battlefield_values[i]) * probability_player1_wins
        return score        

def generate_pure_strategies(units_to_distribute: int, battlefield_count: int) -> List[Tuple[int]]:
    """
        Generate all possible pure strategies for the given number of units to distribute and battlefield count.
    """
    return [strategy for strategy in product(range(units_to_distribute + 1), repeat=battlefield_count) if sum(strategy) == units_to_distribute]

def verify_equilibrium(mixed_strategies: List[str], win_or_score_or_lottery: ScoringSchema, battlefield_values: List[int], tolerance: Decimal) -> bool:
    """
        Given a list of mixed strategies, verify if they are an equilibrium.
        Return "PASSED" if they are, the failing mixed strategy otherwise.
        - mixed_strategies: list of mixed strategies to verify.
        - win_or_score_or_lottery: win, score or lottery.
            - win: maximize the expected number of overall wins under the deterministic scoring scheme 
            (the player who allocates more units to a battlefield wins the points for that battlefield (with the points split evenly in case of a tie, including a 0-0 tie))
            - score: maximize the expected number of overall points under the deterministic scoring scheme
            - lottery: maximize the expected number of overall points under the probabilistic scoring scheme (the player with the most units in a particular battlefield a higher 
            probability of winning the points for that battlefield: if player 1 has allocated u_1 units in a battlefield, 
            and player 2 has allocated u_2 units, then the probability that player 1 wins the battlefield and all the points for it is (u_1^2/u_1^2 + u_2^2), 
            and the probability that player 2 wins the battlefield and all the points for it is (u_2^2/u_1^2 + u_2^2). (If u_1 = u_2 = 0 then the winner of the battlefield is determined by a fair coin flip.)
        - battlefield_values: list of values for each battlefield.
        - tolerance: tolerance for the comparison of the expected payoffs.
    """
    proposed_equilibrium_mixed_strategy = [parse_pure_strategy(strategy) for strategy in mixed_strategies]
    proposed_equilibrium_mixed_strategy_payoff = sum(probability * probability * compute_strategy_payoff(pure_strategy, win_or_score_or_lottery, battlefield_values) for pure_strategy, probability in proposed_equilibrium_mixed_strategy)
    units_to_distribute = sum(proposed_equilibrium_mixed_strategy[0][0])
    for pure_strategy_deviation in generate_pure_strategies(units_to_distribute, len(battlefield_values)):
        # Check if player 1's unilateral deviation is profitable
        player1_unilateral_deviation = sum(probability * compute_strategy_payoff(pure_strategy_deviation, win_or_score_or_lottery, battlefield_values, pure_strategy_in_mixed_strategy) for pure_strategy_in_mixed_strategy, probability in proposed_equilibrium_mixed_strategy)
        if player1_unilateral_deviation > proposed_equilibrium_mixed_strategy_payoff + tolerance:
            print(f'E[{pure_strategy_deviation}, Y] = {player1_unilateral_deviation} > {proposed_equilibrium_mixed_strategy_payoff}')
            return False
        # Check if player 2's unilateral deviation is profitable
        player2_unilateral_deviation = sum(probability * compute_strategy_payoff(pure_strategy_in_mixed_strategy, win_or_score_or_lottery, battlefield_values, pure_strategy_deviation) for pure_strategy_in_mixed_strategy, probability in proposed_equilibrium_mixed_strategy)
        if player2_unilateral_deviation < proposed_equilibrium_mixed_strategy_payoff - tolerance:
            print(f'E[X, {pure_strategy_deviation}] = {player2_unilateral_deviation} < {proposed_equilibrium_mixed_strategy_payoff}')
            return False
    print('PASSED')
    return True
        
    
def find_equilibrium(win_or_score_or_lottery: str, units_to_distribute: int, battlefield_values: List[int], tolerance: Decimal) -> List[Tuple[int, int, int, str]]:
    """
        Find the equilibrium for the given battlefield values.
        Return a list of tuples with the units distributed for each battlefield.
    """
    pass

if __name__ == '__main__':
    getcontext().prec = 18
    random.seed(19)
    find_or_verify, tolerance, win_or_score_or_lottery, units_to_distribute, battlefield_values, mixed_strategies_to_verify = parse_args()
    
    if find_or_verify == 'verify':
        verify_equilibrium(mixed_strategies_to_verify, win_or_score_or_lottery, battlefield_values, tolerance)
    else:
        find_equilibrium(win_or_score_or_lottery, units_to_distribute, battlefield_values, tolerance)

   

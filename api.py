import random

MAX_ROLLS_PER_TURN = 2
"""int: Maximum additional rolls allowed per turn (excluding the first roll)"""

ACTION_INDEX_LIMIT = 32
"""int: Threshold for distinguishing dice reroll actions from scorecard row selection actions."""

THREE_OF_A_KIND_SCORE = lambda dice: sum(dice)
"""Lambda: Calculates the score for 'Three of a Kind' by summing the dice values."""

FOUR_OF_A_KIND_SCORE = lambda dice: sum(dice)
"""Lambda: Calculates the score for 'Four of a Kind' by summing the dice values."""

FULL_HOUSE_SCORE = lambda dice: 25
"""Lambda: Fixed score of 25 for a 'Full House' combination."""

SMALL_STRAIGHT_SCORE = lambda dice: 30
"""Lambda: Fixed score of 30 for a 'Small Straight' combination."""

LARGE_STRAIGHT_SCORE = lambda dice: 40
"""Lambda: Fixed score of 40 for a 'Large Straight' combination."""

YAHTZEE_SCORE = lambda dice: 50
"""Lambda: Fixed score of 50 for a 'Yahtzee' (all dice showing the same value)."""

CHANCE_SCORE = lambda dice: sum(dice)
"""Lambda: Calculates the score for 'Chance' by summing all dice values."""

ENABLE_BONUS = True
"""Boolean: Whether or not to enable the bonus for the Upper Section of the board."""

BONUS_THRESHOLD_POINTS = 42
"""Int: Requirement of the total points scored in the Upper Section of the board to acquire a bonus."""

BONUS_SCORE = 35
"""Int: Fixed bonus score for reaching the given threshold of points in the Upper Section of the board."""


class YahtzeeGame:
    def __init__(self):
        """Initialize a new Yahtzee game."""
        self.number_mapping = {
            'ones': 1,
            'twos': 2,
            'threes': 3,
            'fours': 4,
            'fives': 5,
            'sixes': 6
        }
        self.scores = [
            'ones',
            'twos',
            'threes',
            'fours',
            'fives',
            'sixes',
            'three_of_a_kind',
            'four_of_a_kind',
            'full_house',
            'small_straight',
            'large_straight',
            'yahtzee',
            'chance',
        ]
        self.newGame()

    def newGame(self):
        """Start a new game, resetting all attributes to their initial state."""
        self.dice = [0] * 5  # Values of the 5 dice
        self.rolls_left = 1  # Rolls remaining for the turn
        self.rollDice()
        self.rolls_left = MAX_ROLLS_PER_TURN  # Rolls remaining for the turn
        self.scorecard = {k: None for k in self.scores}
        self.completed_rows = [False] * len(self.scorecard)  # Flags for completed rows
        self.last_reward = 0
        self.total_reward = 0
        if ENABLE_BONUS:
            self.bonus = False
        self.finished = False

    def rollDice(self, reroll_flags=None):
        """Roll the dice, rerolling specific dice based on the flags provided.

        Args:
            reroll_flags (list): A list of 5 integers (1 to reroll, 0 to keep). If None, all dice are rerolled.

        Raises:
            Exception: If no rolls are left for this turn.
        """
        if self.rolls_left == 0:
            raise Exception('No rolls left for this turn.')

        if reroll_flags is None:
            reroll_flags = [1] * 5  # Reroll all dice if no flags provided

        for i in range(5):
            if reroll_flags[i] == 1:
                self.dice[i] = random.randint(1, 6)

        self.rolls_left -= 1

    def rollLeft(self):
        """Get the number of rolls left in the current turn.

        Returns:
            int: Number of rolls left.
        """
        return self.rolls_left

    def getDiceValues(self):
        """Get the current values of the dice.

        Returns:
            list: A copy of the current dice values.
        """
        return [v for v in self.dice]  # Secure copy

    def chooseRow(self, row):
        """Choose a row on the scorecard to complete and calculate the score for it.

        Args:
            row (str): The name of the row to complete.

        Raises:
            Exception: If the row has already been used or is invalid.
        """
        if self.scorecard.get(row) is not None:
            raise Exception(f'Row {row} has already been used.')

        if row in self.number_mapping:
            num = self.number_mapping[row]
            score = self.dice.count(num) * num
        elif row == 'three_of_a_kind':
            if any(self.dice.count(x) >= 3 for x in set(self.dice)):
                score = THREE_OF_A_KIND_SCORE(self.dice)
            else:
                score = 0
        elif row == 'four_of_a_kind':
            if any(self.dice.count(x) >= 4 for x in set(self.dice)):
                score = FOUR_OF_A_KIND_SCORE(self.dice)
            else:
                score = 0
        elif row == 'full_house':
            if sorted([self.dice.count(x) for x in set(self.dice)]) == [2, 3]:
                score = FULL_HOUSE_SCORE(self.dice)
            else:
                score = 0
        elif row == 'small_straight':
            if (
                    {1, 2, 3, 4}.issubset(self.dice) or
                    {2, 3, 4, 5}.issubset(self.dice) or
                    {3, 4, 5, 6}.issubset(self.dice)
            ):
                score = SMALL_STRAIGHT_SCORE(self.dice)
            else:
                score = 0
        elif row == 'large_straight':
            if sorted(self.dice) in [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]]:
                score = LARGE_STRAIGHT_SCORE(self.dice)
            else:
                score = 0
        elif row == 'yahtzee':
            if len(set(self.dice)) == 1:
                score = YAHTZEE_SCORE(self.dice)
            else:
                score = 0
        elif row == 'chance':
            score = CHANCE_SCORE(self.dice)
        else:
            raise Exception(f'Invalid row: {row}')

        self.scorecard[row] = score
        self.last_reward = score
        self.total_reward += score

        row_index = list(self.scorecard.keys()).index(row)
        self.completed_rows[row_index] = True

        if ENABLE_BONUS:
            if (not self.bonus) and (row in self.number_mapping):
                upper_section_total = sum([
                    points
                    for score in self.number_mapping.keys()
                    if (points := self.scorecard[score]) is not None
                ])
                if upper_section_total >= BONUS_THRESHOLD_POINTS:
                    self.last_reward += BONUS_SCORE
                    self.total_reward += BONUS_SCORE
                    self.bonus = True

        self.rolls_left = 1
        self.rollDice()
        self.rolls_left = MAX_ROLLS_PER_TURN

        if all(v is not None for v in self.scorecard.values()):
            self.finished = True

    def chooseAction(self, action):
        """Perform an action: either reroll dice or complete a row.

        Args:
            action (int): The action to perform. Actions < ACTION_INDEX_LIMIT are rerolls, others are row selections.
        """
        if action < ACTION_INDEX_LIMIT:
            reroll_flags = [(action >> i) & 1 for i in range(5)]
            self.rollDice(reroll_flags)
        else:
            row = list(self.scorecard.keys())[action - ACTION_INDEX_LIMIT]
            self.chooseRow(row)

    def getLastReward(self):
        """Get the score of the last completed row.

        Returns:
            int: Score of the last completed row.
        """
        return self.last_reward

    def getTotalReward(self):
        """Get the total score for the game.

        Returns:
            int: Total score.
        """
        return self.total_reward

    def hasFinished(self):
        """Check if the game is finished.

        Returns:
            bool: True if the game is finished, False otherwise.
        """
        return self.finished

    def getAllRows(self):
        """Get a list of all available rows.

        Returns:
            list: Names of all rows.
        """
        return [row for row in self.scores]

    def getCompletedRows(self):
        """Get a list of completed rows.

        Returns:
            list: Names of completed rows.
        """
        return [row for row, score in self.scorecard.items() if score is not None]


# Example usage of the YahtzeeGame API
if __name__ == '__main__':
    game = YahtzeeGame()
    game.newGame()

    # Perform some example actions
    game.rollDice()
    dice_turn_history = [game.getDiceValues()]
    print('Dice after the first roll:', dice_turn_history[-1])

    dices_to_roll = [1, 0, 0, 1, 1]
    game.rollDice(dices_to_roll)
    dice_turn_history.append(game.getDiceValues())
    for idx, to_roll in enumerate(dices_to_roll):
        if not to_roll:
            assert dice_turn_history[-2][idx] == dice_turn_history[-1][idx]
    print('Dice after the second roll:', dice_turn_history[-1])

    try:
        dices_to_roll = [0, 0, 1, 1, 1]
        game.rollDice(dices_to_roll)
    except Exception as e:
        assert str(e) == 'No rolls left for this turn.'

    game.chooseRow('fours')

    dice_turn_history = [game.getDiceValues()]
    print('Dice after the first roll:', dice_turn_history[-1])

    dices_to_roll = [0, 0, 1, 1, 1]
    game.rollDice(dices_to_roll)
    dice_turn_history.append(game.getDiceValues())
    for idx, to_roll in enumerate(dices_to_roll):
        if not to_roll:
            assert dice_turn_history[-2][idx] == dice_turn_history[-1][idx]
    print('Dice after the second roll:', game.getDiceValues())

    game.chooseRow('fives')

    print('Last reward:', game.getLastReward())
    print('Total reward:', game.getTotalReward())
    print('Completed rows:', game.getCompletedRows())
    print('Game finished?', game.hasFinished())

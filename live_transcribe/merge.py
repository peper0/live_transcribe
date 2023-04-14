import numpy as np

def merge_strings(first_seq, second_seq):
    """
    Merge two overlapping strings.
    :param first_seq:
    :param second_seq:
    :return:

    >>> merge_strings("one two three fourx ", "three four five six seven")
    'one two three four five six seven'
    >>> merge_strings("aaa", "bbb")
    'aaabbb'
    >>> merge_strings("aaa", "aa")
    'aaa'
    >>> merge_strings("aa", "")
    'aa'
    >>> merge_strings("", "aa")
    'aa'
    >>> merge_strings("", "")
    ''
    >>> merge_strings("aabbaa", "aaccaa")
    'aabbaaccaa'
    >>> merge_strings(["aa", "bb"], ["bb", "cc"])
    ['aa', 'bb', 'cc']
    """
    DEC_FIRST = 0
    DEC_SECOND = 1
    DEC_NONE = 2
    DEC_BOTH = 3

    cost = np.zeros((len(first_seq) + 1, len(second_seq) + 1), dtype=np.int32)
    decision = np.zeros((len(first_seq) + 1, len(second_seq) + 1), dtype=np.int32)
    # cost[i, j] = cost of transforming earlier_seq[:i] into later_seq[:j]
    # later_seq should start inside earlier_seq, so cost for i>=0 and  j==0 is low
    # but cost for j>0 and i==0 is large
    # INF_COST = 1000
    REPLACE_COST = 4
    DELETE_MIDDLE_COST = 4
    UNPREFIX_FIRST_COST = 1
    UNPREFIX_SECOND_COST = 2
    UNSUFFIX_FIRST_COST = 2
    UNSUFFIX_SECOND_COST = 1
    cost[1:, 0] = UNPREFIX_FIRST_COST * np.arange(1, len(first_seq) + 1)
    decision[1:, 0] = DEC_FIRST
    cost[0, 1:] = UNPREFIX_SECOND_COST * np.arange(1, len(second_seq) + 1)
    decision[0, 1:] = DEC_SECOND

    for i in range(1, len(first_seq) + 1):
        for j in range(1, len(second_seq) + 1):
            candidates = [
                (cost[i - 1, j - 1] + REPLACE_COST, DEC_NONE),
                (cost[i - 1, j] + (DELETE_MIDDLE_COST if j < len(second_seq) else UNSUFFIX_FIRST_COST), DEC_FIRST),
                (cost[i, j - 1] + (DELETE_MIDDLE_COST if i < len(first_seq) else UNSUFFIX_SECOND_COST), DEC_SECOND),
            ]
            if first_seq[i - 1] == second_seq[j - 1]:
                cost[i, j] = cost[i - 1, j - 1]
                decision[i, j] = DEC_BOTH
                candidates.append((cost[i - 1, j - 1], DEC_BOTH))

            # if i == len(first_seq):
            #     print(f"i={i}, j={j}, candidates={candidates}")
            cost[i, j], decision[i, j] = min(candidates, key=lambda x: x[0])


    def backtrack(i, j):
        if i == 0 and j == 0:
            return None
        else:
            d = decision[i, j]
            if d == DEC_FIRST:
                prev = backtrack(i - 1, j)
                letter = first_seq[i - 1:i] if j == 0 else None
            elif d == DEC_SECOND:
                prev = backtrack(i, j - 1)
                letter = second_seq[j - 1:j] if i == len(first_seq) else None
            elif d == DEC_BOTH:
                prev = backtrack(i - 1, j - 1)
                letter = first_seq[i - 1:i]
            else:
                prev = backtrack(i - 1, j - 1)
                if j < len(second_seq) - i:  # near ends there are greater risk of mistakes
                    letter = first_seq[i - 1:i]
                else:
                    letter = second_seq[j - 1:j]
            # print(f"{letter} {i} {j} {d} {cost[i, j]}")

            if letter is None:
                return prev
            elif prev is None:
                return letter
            else:
                return prev + letter


    res = backtrack(len(first_seq), len(second_seq))
    if res is None:
        res = type(first_seq)()
    return res


def split_into_lines(text: str, max_line_length: int):
    """
    Split text into lines of max_line_length characters.
    :param text:
    :param max_line_length:
    :return:

    >>> split_into_lines("one two three four five six seven", 10)
    ['one two', 'three four', 'five six', 'seven']
    >>> split_into_lines("123456789012 1 123456789012 32 43", 10)
    ['123456789012', '1', '123456789012', '32 43']


    """
    lines = []
    line = ""
    for word in text.split(" "):
        if len(line) + len(word) > max_line_length and line:
            lines.append(line)
            line = word
        else:
            if line:
                line += " "
            line += word
    if line:
        lines.append(line)
    return lines
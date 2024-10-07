from sys import argv
from decimal import Decimal, getcontext

def parse_args():
    find_or_verify = argv[1][2:]
    tolerance = Decimal('1e-6')
    units_to_distribute = None
    if argv[2] == '--tolerance':
        tolerance = Decimal(argv[3])
        win_or_score_or_lottery = argv[4][2:]
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
    return find_or_verify, tolerance, win_or_score_or_lottery, units_to_distribute, battlefield_values


if __name__ == '__main__':
    getcontext().prec = 18
    find_or_verify, tolerance, win_or_score_or_lottery, units_to_distribute, battlefield_values = parse_args()
    print(find_or_verify, tolerance, win_or_score_or_lottery, units_to_distribute, battlefield_values)
   

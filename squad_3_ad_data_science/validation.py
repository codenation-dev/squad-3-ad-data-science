def custom_apk(market_list, test_list, k=10000):

    '''
        @brief Custom function for the average precision at k - AP@k

        @param market_list: list of ids from all the recommended companies
        sorted by descending score (order does matter)

        @param test_list: list of ids from all companies in the ~20% test
        dataframe (order does not matter)

        @param k: number of recommendations

        @return score of test
    '''

    if len(market_list) > k:
        market_list = market_list[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(market_list):
        if p in test_list:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / len(test_list)

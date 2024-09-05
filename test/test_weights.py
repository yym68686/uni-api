def weighted_round_robin(weights):
    provider_names = list(weights.keys())
    current_weights = {name: 0 for name in provider_names}
    num_selections = total_weight = sum(weights.values())
    weighted_provider_list = []

    for _ in range(num_selections):
        max_ratio = -1
        selected_letter = None

        for name in provider_names:
            current_weights[name] += weights[name]
            ratio = current_weights[name] / weights[name]

            if ratio > max_ratio:
                max_ratio = ratio
                selected_letter = name

        weighted_provider_list.append(selected_letter)
        current_weights[selected_letter] -= total_weight

    return weighted_provider_list

# 权重和选择次数
weights = {'a': 5, 'b': 3, 'c': 2}
index = {'a', 'c'}

result = dict(filter(lambda item: item[0] in index, weights.items()))
print(result)
# result = {k: weights[k] for k in index if k in weights}
# print(result)
weighted_provider_list = weighted_round_robin(weights)
print(weighted_provider_list)

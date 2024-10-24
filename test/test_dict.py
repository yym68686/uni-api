a = [
    {"a": 1, "b": 2, "c": 3},
    {"a": 4, "b": 5, "c": 6},
    {"a": 7, "b": 8, "c": 9}
]
import copy
for item in a:
    new_item = copy.deepcopy(item)
    new_item["a"] = 10
    del new_item["b"]
    # print(item)
print(a)

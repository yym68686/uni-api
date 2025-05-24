a = []
for i in a:
    if not i['balance'].startswith('-') and float(i['balance']) > 0:
        print(i['key'])

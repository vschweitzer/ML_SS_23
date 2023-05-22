def fn(a, b, c, d, e, f):
    return ((a and b) or (c and d) and (bool(e) != bool(f))) or (a != b)


params = []
results = []

for a in [0, 1]:
    for b in [0, 1]:
        for c in [0, 1]:
            for d in [0, 1]:
                for e in [0, 1]:
                    for f in [0, 1]:
                        params.append([a, b, c, d, e, f])
                        results.append(int(fn(a, b, c, d, e, f)))

print(params)
print()
print(results)

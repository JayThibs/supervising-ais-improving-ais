from ast import literal_eval

def round_list(l, n):
    if isinstance(l[0], list):
        return [round_list(x, n) for x in l]
    else:
        return [round(x, n) for x in l]

def literal_eval_fallback(s, default_value):
    try:
        return literal_eval(s)
    except:
        print(f"Could not parse {s}")
        return default_value
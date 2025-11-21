from typing import List, Union

Number = Union[int, float]

def parse_number_list(src: str) -> List[Number]:
    """
    Parse a string representing a Python-like list of numbers into a Python list.
    Supported numbers: ints and floats (incl. scientific notation).
    Examples accepted: [1, -2, 3.0, .5, 6., 1e3, -2.5E-2], [], [1,2,]
    """
    i, n = 0, len(src)

    def err(msg: str) -> None:
        # Show a small window around the error location
        start = max(0, i - 10)
        end = min(n, i + 10)
        snippet = src[start:end].replace("\n", "\\n")
        raise ValueError(f"{msg} at position {i}. Context: '{snippet}'")

    def skip_ws() -> None:
        nonlocal i
        while i < n and src[i].isspace():
            i += 1

    def parse_number() -> Number:
        nonlocal i

        num_str = ""

        # Optional sign
        if i < n and src[i] in "+-":
            num_str += src[i]
            i += 1

        # Integer / fraction part
        digits_before = 0
        while i < n and src[i].isdigit():
            num_str += src[i]
            i += 1
            digits_before += 1

        # Optional decimal point + fractional digits
        if i < n and src[i] == '.':
            num_str += '.'
            i += 1
            digits_after = 0
            while i < n and src[i].isdigit():
                num_str += src[i]
                i += 1
                digits_after += 1
            if digits_before + digits_after == 0:
                err("Invalid float literal; expected digits before or after '.'")
        elif digits_before == 0:
            # Neither digits nor '.' -> invalid start of number
            err("Expected a digit or '.' when parsing a number")

        # Optional exponent
        if i < n and src[i] in 'eE':
            num_str += src[i]
            i += 1
            if i < n and src[i] in '+-':
                num_str += src[i]
                i += 1
            exp_digits = 0
            while i < n and src[i].isdigit():
                num_str += src[i]
                i += 1
                exp_digits += 1
            if exp_digits == 0:
                err("Invalid exponent; expected digits after 'e'/'E'")

        # Convert to int when no '.' or exponent; otherwise float
        try:
            if '.' in num_str or 'e' in num_str.lower():
                return float(num_str)
            else:
                return int(num_str)
        except Exception:
            err(f"Failed to convert {num_str!r} to a number")

    # --- Parse the list ---
    skip_ws()
    if i >= n or src[i] != '[':
        err("Expected '[' to start a list")
    i += 1

    result: List[Number] = []

    while True:
        skip_ws()
        if i < n and src[i] == ']':  # empty list or end of list
            i += 1
            break

        # Parse one number
        value = parse_number()
        result.append(value)

        skip_ws()
        if i < n and src[i] == ',':
            i += 1  # move past comma
            # allow trailing comma: next iteration may immediately see ']'
            continue
        elif i < n and src[i] == ']':
            i += 1
            break
        elif i >= n:
            err("Unexpected end of input; expected ',' or ']' after a number")
        else:
            err("Expected ',' or ']' after a number")

    # No trailing junk allowed (except whitespace)
    skip_ws()
    if i != n:
        err("Unexpected characters after the closing ']'")

    return result

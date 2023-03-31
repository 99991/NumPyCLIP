# OpenAI's tokenizer uses the regex library instead of Python's re module.
# I believe that the only feature of the regex library that has been used are
# Unicode property shortcuts (\p{N} and \p{L}).
# This file extracts all unicode ranges which match \p{N} or \p{L}.
import re, regex, sys

for name, target_pattern in [
    ("pN.txt", r"[\p{N}]"),
    ("pL.txt", r"[\p{L}]"),
]:
    starts = []
    ends = []
    for i in range(sys.maxunicode):
        try:
            s = chr(i)
            if regex.match(target_pattern, s):

                if starts and ends[-1] + 1 == i:
                    ends[-1] = i
                else:
                    starts.append(i)
                    ends.append(i)
        except UnicodeEncodeError:
            pass

    pattern = "".join(f"{chr(a)}-{chr(b)}" for a, b in zip(starts, ends))

    print(pattern)

    with open(name, "w", encoding="utf-8") as f:
        f.write(pattern)

    # Double-check correctness
    for i in range(sys.maxunicode):
        try:
            s = chr(i)

            m1 = regex.match(target_pattern, s)
            m2 = re.match(pattern, s)
            assert bool(m1) == bool(m2), f"ERROR: '{s}' {hex(ord(s))} {m1} {m2}"

        except UnicodeEncodeError:
            pass

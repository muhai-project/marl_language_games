def write_measure(monitor, fname):
    """Writes data out to a specified file with a s-expression format required by Babel."""
    out = ""
    for serie in monitor:
        if isinstance(serie[0], bool) or isinstance(serie[0], int):
            data = str([int(i) for i in serie])
        else:
            data = str([round(i, ndigits=2) for i in serie])
        # remove commas and square brackets, add brackets around list and concatenate lists
        data = data.replace(",", "").replace("[", "").replace("]", "")
        data = "(" + data + ")"
        out += data
    out = "((" + out + "))"  # add final round brackets

    with open(f"{fname}.lisp", "w") as file:
        file.write(out)


def write_measure_competition(monitor, fname):
    """Writes competition data out to a specified file with a s-expression format required by Babel."""
    out = ""
    for key, vals in monitor.items():
        vals = list(reversed(vals))
        data = ""
        print("VAAAALS: ", len(vals))
        for val in vals:
            if isinstance(val, bool) or isinstance(val, int):
                data += str(int(val)) + " "
            elif isinstance(val, str):
                data += val + " "
            else:
                data += str(round(val, ndigits=2)) + " "
        data = data[:-1]

        # remove commas and square brackets, add brackets around list and concatenate lists
        data = f"({key} (NIL ({data})))"

        out += data
    out = "((" + out + "))"  # add final round brackets

    with open(f"{fname}.lisp", "w") as file:
        file.write(out)

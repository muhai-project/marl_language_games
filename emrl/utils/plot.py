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

def write_measure(monitor, fname):
    """Writes data out to a specified file with a s-expression format required by Babel."""
    out = ""
    for serie in monitor:
        data = str(
            [int(i) for i in serie]
        )  # change booleans to integers (e.g. comm. success)
        data = (
            data.replace(",", "").replace("[", "").replace("]", "")
        )  # remove commas and square brackets
        data = "(" + data + ")"  # add brackets around list
        out += data  # concatenate strings
    out = "((" + out + "))"  # add final round brackets

    with open(f"{fname}.lisp", "w") as file:
        file.write(out)

def convert_monitor(monitor):
    """Converts monitor to a string (into s-expression format)

    Args:
        monitor (list): a list of data to be converted to a string

    Returns:
        str: s-expression version of the data inside the given monitor
    """
    out = ""
    for trial in monitor:
        if isinstance(trial[0], bool) or isinstance(trial[0], int):
            data = str([int(i) for i in trial])
        else:
            data = str([round(i, ndigits=2) for i in trial])
        # remove commas and square brackets, add brackets around list and concatenate lists
        data = data.replace(",", "").replace("[", "").replace("]", "")
        data = "(" + data + ")"
        out += data
    out = "((" + out + "))"  # add final round brackets

    return out


def write_measure(monitor, fname):
    """Converts and writes data out to a file with a given filename."""

    data = convert_monitor(monitor)
    with open(f"{fname}.lisp", "w") as file:
        file.write(data)


def write_measure_competition(monitor, fname):
    """Writes competition data out to a specified file (into s-expression format)."""
    out = ""
    for key, vals in monitor.items():
        data = ""
        for val in vals:
            if isinstance(val, bool) or isinstance(val, int):
                data += str(int(val)) + " "
            elif isinstance(val, str):
                data += val + " "
            else:
                data += str(round(val, ndigits=2)) + " "
        data = data[:-1]

        # remove commas and square brackets, add brackets around list and concatenate lists
        data = f"({key} ({data}))"

        out += data
    out = "((" + out + "))"  # add final round brackets

    with open(f"{fname}.lisp", "w") as file:
        file.write(out)

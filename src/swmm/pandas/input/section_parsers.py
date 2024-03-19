from __future__ import annotations


from typing import Callable


def _coerce_float(data):
    try:
        return float(data)
    except ValueError:
        return data


def _strip_comment(line: str):
    try:
        return line[: line.index(";")], line[line.index(";") :]

    except ValueError:
        return line, ""


def _is_line_comment(line: str):
    try:
        return line.strip()[0] == ";"
    except IndexError:
        return False


def _is_data(line: str):
    if len(line) == 0 or line.strip()[0:2] == ";;" or line.strip()[0] == "[":
        return False
    return True


def _default_assigner(line: list):
    return line


def _default_parser(text: str, ncols: int, assigner: Callable = _default_assigner):
    rows = text.split("\n")
    data = []
    line_comment = ""
    for row in rows:
        if not _is_data(row):
            line_comment += row
            continue

        elif len(row) == 0:
            continue

        line, comment = _strip_comment(row)
        line_comment += comment

        row_data = [""] * (ncols + 1)
        split_data = [_coerce_float(val) for val in row.split()]
        row_data[:ncols] = assigner(split_data)
        row_data[-1] = line_comment
        data.append(row_data)
        line_comment = ""

    return data


def _curves_parser(text: str, ncols: int):
    rows = text.split("\n")
    data = []
    line_comment = ""

    pump = ""
    typ = ""
    coords = []
    for row in rows:
        if not _is_data(row):
            # line_comment += row
            continue

        elif len(row) == 0:
            continue

        split_data = [_coerce_float(val) for val in row.split()]
        # print(split_data)
        if pump != split_data[0]:
            if len(coords) > 0:
                data.append([pump, typ, coords, line_comment])

            pump = split_data[0]
            typ = split_data[1]
            coords = []
            line, comment = _strip_comment(row)
            line_comment = comment
            coord_data = split_data[2:]
            # print(coord_data)

        else:
            coord_start_idx = 2 - len(split_data) % 2
            coord_data = split_data[coord_start_idx:]

        coords += [
            [coord_data[i], coord_data[i + 1]] for i in range(0, len(coord_data), 2)
        ]
    data.append([pump, typ, coords, line_comment])
    return data


# def _outfalls_parser(text:str,ncols:int):
#     def assigner(line:list):
#         if len(split_data) == 5:
#             row_data[0:3] = split_data[0:3]
#             row_data[4:] = split_data[4:6]
#         elif len(split_data)

#     rows = text.split('\n')
#     for row in rows:
#         if len(row) ==0 or row.strip()[0]==';':
#             continue

#         row_data = [""] * ncols
#         split_data = [_coerce_float(val) for val in row.split()]
#         if len(split_data) == 5:
#             row_data[0:3] = split_data[0:3]
#             row_data[4:] = split_data[4:6]
#         elif len(split_data)

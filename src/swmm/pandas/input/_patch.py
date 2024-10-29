from swmm.pandas.input._section_classes import SectionBase, _sections
import re


def split_patch(inp):
    drops = []
    keeps = []
    _section_re = re.compile(R"^\[[\s\S]*?(?=^\[|\Z)", re.MULTILINE)
    _section_keys = tuple(_sections.keys())
    with open(inp) as f:
        text: str = f.read()
    for section in _section_re.findall(text):
        name: str = re.findall(R"^\[(.*)\]", section)[0]
        if name.strip().startswith("-"):
            section = section.replace(f"[{name}]", f"[{name.replace('-','')}]")
            outlist = drops
        else:
            outlist = keeps

        outlist.append(section)
    print("\n\n".join(drops))
    print("\n\n".join(keeps))

    # if
    #  try:
    #     section_idx = list(
    #         name.lower().startswith(x.lower()) for x in _sections
    #     ).index(True)
    #     section_key = self._section_keys[section_idx]
    #     self._section_texts[section_key] = data
    # except Exception as e:
    #     logger.error(f"Error parsing section: {name}")
    #     raise e

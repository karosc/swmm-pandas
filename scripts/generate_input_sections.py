from swmm.pandas.input._section_classes import _sections as sections

out_str = """\

#region
# This section is autgenerated by scripts/generate_input_sections.py

"""

print(len(sections))
for section, obj in sections.items():
    sectstring = f"{obj.__name__.lower()}: sc.{obj.__name__}\n"
    if hasattr(obj, "headings"):
        sectstring += f'"{obj.headings}"\n'
    out_str += sectstring
print(out_str)

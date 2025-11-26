from __future__ import annotations


from typing import TypeVar, Union, overload
import copy
import re
from io import StringIO
from pathlib import Path

import yaml

from swmm.pandas.input import InputFile
from swmm.pandas.input._section_classes import _sections
from swmm.pandas.input.model import Input


class InpPatch:
    def __init__(self, inp_patch: str | Path):
        self._inp_patch = inp_patch
        self.split_patch()
        self.params, self.inp_text = self.parse_frontmatter()

    def parse_frontmatter(self) -> tuple[dict, str]:
        """
        Extract YAML front matter from text content and return both the parsed YAML
        and the remaining content.

        Args:
            content (str): Text content that may contain YAML front matter

        Returns
        -------
            tuple: (dict, str) - (parsed YAML as dict, content without front matter)
                    If no front matter exists, returns ({}, original content)
        """
        # Pattern to match YAML front matter between --- delimiters
        with open(self._inp_patch) as f:
            content: str = f.read()
        pattern = r"^---\n((.|\n)*?)\n---$"
        match = re.match(pattern, content, re.MULTILINE)

        if not match:
            return {}, content

        try:
            # Parse YAML content from the first capture group
            yaml_dict = yaml.safe_load(match.group(1))
            # Get remaining content from the second capture group
            remaining_content = match.group(2)
            return yaml_dict, remaining_content
        except yaml.YAMLError:
            # Return empty dict and original content if YAML parsing fails
            return {}, content

    def split_patch(self):

        drops = []
        keeps = []
        _section_re = re.compile(R"^\[[\s\S]*?(?=^\[|\Z)", re.MULTILINE)
        _section_keys = tuple(_sections.keys())
        with open(self._inp_patch) as f:
            text: str = f.read()
        for section in _section_re.findall(text):
            name: str = re.findall(R"^\[(.*)\]", section)[0]
            if name.strip().startswith("-"):
                section = section.replace(f"[{name}]", f"[{name.removeprefix('-')}]")
                outlist = drops
            else:
                outlist = keeps

            outlist.append(section)
        self._drop_inp_str = "\n\n".join(drops)
        self._patch_inp_str = "\n\n".join(keeps)

        self._drop_inp_obj = InputFile(StringIO(self._drop_inp_str))

        self._patch_inp_obj = InputFile(StringIO(self._patch_inp_str))

    @overload
    def patch(self, inp: Input) -> Input: ...

    @overload
    def patch(self, inp: InputFile) -> InputFile: ...

    def patch(self, inp: Input | InputFile) -> Input | InputFile:
        if isinstance(inp, Input):
            inp._sync()
            _inp = copy.deepcopy(inp._inp)
        else:
            _inp = copy.deepcopy(inp)

        for section_name, section_class in _sections.items():
            public_property_name = section_class.__name__.lower()
            # print(public_property_name)
            getattr(_inp, public_property_name)._drop(
                getattr(self._drop_inp_obj, public_property_name),
            )
            getattr(_inp, public_property_name)._patch(
                getattr(self._patch_inp_obj, public_property_name),
            )

        if isinstance(inp, Input):
            return Input(_inp)
        else:
            return _inp

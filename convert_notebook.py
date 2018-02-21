#!/bin/env python3
import os.path
import os
import nbformat
import re
import shutil
import glob
import sys
from os.path import basename, dirname
import subprocess

"""Convert jupyter notebooks to hugo posts.
Usage: specify the directory in which you store your jupyter notebooks in
SEARCH_DIR. It is assumed that all notebooks are contained within their
own folder, and that the foldername starts with the publish date in the
form 2018-02-20. The complete folder name will be used as the slug of
the post. Set OUTPUT_DIR to the directory in content where you wish to
post your content to.
"""
BASE_DIR = "/home/ocius/Repos/fbehrens.com/"
SEARCH_DIR = "notebooks/"
OUTPUT_DIR = "content/posts/"

def render_notebook(to_render):
    os.chdir(BASE_DIR)
    nb_dir = basename(dirname(to_render))
    markdown_file = os.path.join(OUTPUT_DIR, nb_dir + ".md")
    resources_dir = os.path.join(OUTPUT_DIR, nb_dir)
    if os.path.isdir(resources_dir):
        shutil.rmtree(resources_dir)
    os.makedirs(resources_dir)

    with open(to_render) as input:
        notebook = nbformat.reads(input.read(), as_version=4)

    front_matter = "---\n"

    m = re.search(r'# *(.*)\n', notebook.cells[0].source, re.M)
    title = m.group(1)
    front_matter += f"title: \"{title}\"\n"
    notebook.cells[0].source = notebook.cells[0].source.replace(m.group(0), "")

    publish_date = nb_dir[:10]
    front_matter += f"date: {publish_date}\n"
    front_matter += "---\n"

    inline_math = re.compile(r'(?:[^\$]|^)\$[^\$]+\$(?:[^\$]|$)')
    multiline_math = re.compile(r'\$\$[^\$]+\$\$')

    for i in range(len(notebook.cells)):
        cell = notebook.cells[i]
        if not cell['cell_type'] == 'markdown':
            continue
        source = cell['source']

        inlines = inline_math.findall(source)
        for inline in inlines:
            r = inline.replace(r"\\", r"\\\\\\\\")
            r = r.replace("_", r"\_")
            source = source.replace(inline, r)

        multilines = multiline_math.findall(source)
        for multiline in multilines:
            r = multiline.replace(r"\\", r"\\\\\\\\")
            r = r.replace("_", r"\_")
            source = source.replace(multiline, r)

        cell['source'] = source

    from nbconvert import MarkdownExporter
    md_exporter = MarkdownExporter()
    body, resources = md_exporter.from_notebook_node(notebook)

    files = resources['outputs']
    for filename in files:
        p = os.path.join(resources_dir, filename)
        with open(p, "wb") as f:
            f.write(files[filename])

    with open(markdown_file, "w") as output:
        output.write(front_matter)
        output.write(body)

if __name__ == "__main__":
    os.chdir(BASE_DIR)
    try:
        subprocess.run('test -z "$(git status --porcelain)"', shell=True, check=True)
    except:
        print("Warning: your repository is not clean. Are you sure you wish to continue?")
        answer = input("[y/n] >> ")
        if not answer.strip().startswith("y"):
            sys.exit(1)

    if len(sys.argv) == 2:
        to_render = sys.argv[1]
    else:
        notebook_files = list(reversed(sorted(glob.glob("notebooks/*/*.ipynb"))))
        for i, name in enumerate(notebook_files):
            print(f"{i}: {name}")
        i = int(input("Choose which notebook to render: "))
        to_render = notebook_files[i]
    print(f"Rendering {to_render}")
    render_notebook(to_render)

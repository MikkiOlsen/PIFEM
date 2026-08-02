import os
import sys
import numpy as np

from mesh.gmsh import import_msh

# T[:, 8] is the propno indexing rows of the material table H. By default it
# comes from the gmsh physical surfaces, ranked by tag number, so the group
# names are ignored. For more than one material, name them instead:
#     X, T = import_msh(path, materials=['mat_a', 'mat_b'])   # -> propno 1, 2
# which raises if a name is missing rather than swapping your materials.

DEFAULT_MESH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'meshes', 'Simple_plate.msh')


def write_arrays(X, T, out_path):
    with open(out_path, 'w') as f:
        f.write("import numpy as np\n\n")
        f.write("X = np.array(\n")
        f.write(np.array2string(X, separator=',', threshold=np.inf))
        f.write("\n)\n\n")
        f.write("T = np.array(\n")
        f.write(np.array2string(T, separator=',', threshold=np.inf))
        f.write("\n)")


def pick_file():
    from tkinter import filedialog
    return filedialog.askopenfilename(title="Select a .msh file", filetypes=[("Gmsh files", "*.msh")])


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        msh_path = sys.argv[1]
    elif sys.stdin.isatty():
        msh_path = pick_file()
    else:
        msh_path = DEFAULT_MESH

    if not msh_path:
        sys.exit("No mesh selected.")

    X, T = import_msh(msh_path)
    print(f"{msh_path}")
    print(f"nodes:    {X.shape[0]}")
    print(f"elements: {T.shape[0]}")

    out_path = os.path.splitext(msh_path)[0] + '_arrays.py'
    write_arrays(X, T, out_path)
    print(f"wrote: {out_path}")

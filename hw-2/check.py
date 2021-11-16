with open("sponza/sponza.obj", "r") as f:
    vertices = set()
    normals = 0
    positions = 0
    textures = 0
    last_position = 0
    min_position = 1e9
    for i, line in enumerate(f.readlines()):
        if line.startswith("# object"):
            # print(len(vertices) / max(normals, positions, textures, 1), min_position > last_position, min_position, last_position)
            # if not vertices:
            #     print(i)
            vertices = set()
            last_position += positions
            normals = 0
            positions = 0
            textures = 0
            min_position = 1e9
        if line.startswith("vn"):
            normals += 1
        elif line.startswith("v "):
            positions += 1
        elif line.startswith("vt"):
            textures += 1
        if line.startswith("f"):
            vs = line.split()[1:]
            for v in vs:
                vertices.add(v)
                min_position = min(min_position, int(v.split("/")[0]))
                if v.split("/")[2]:
                    print(i)
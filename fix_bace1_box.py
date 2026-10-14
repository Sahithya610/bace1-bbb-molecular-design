import json
box = {"center_x": 30.65, "center_y": 41.54, "center_z": 1.86,
       "size_x": 22.0, "size_y": 22.0, "size_z": 22.0}
with open("docking/bace1_box.json", "w") as f:
    json.dump(box, f, indent=2)
print("Box fixed:", box)

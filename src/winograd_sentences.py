"""Oraciones tipo Winograd usadas en los experimentos de la Fase 1.

No son citas textuales del dataset oficial de Winograd Schema Challenge: son
oraciones propias, del mismo estilo (una ambiguedad de pronombre que se
resuelve segun una sola palabra que cambia entre las dos variantes).
"""

ORACIONES = [
    {
        "nombre": "trophy_suitcase",
        "plantilla": "The trophy doesn't fit in the suitcase because it is too {d}.",
        "candidatos": ["trophy", "suitcase"],
        "variantes": {"big": "trophy", "small": "suitcase"},
    },
    {
        "nombre": "box_drawer",
        "plantilla": "The box doesn't fit in the drawer because it is too {d}.",
        "candidatos": ["box", "drawer"],
        "variantes": {"big": "box", "small": "drawer"},
    },
    {
        "nombre": "bottle_cup",
        "plantilla": "I poured the water from the bottle into the cup until it was {d}.",
        "candidatos": ["bottle", "cup"],
        "variantes": {"empty": "bottle", "full": "cup"},
    },
    {
        "nombre": "knife_bread",
        "plantilla": "The knife couldn't cut the bread because it was too {d}.",
        "candidatos": ["knife", "bread"],
        "variantes": {"dull": "knife", "hard": "bread"},
    },
    {
        "nombre": "man_couch",
        "plantilla": "The man couldn't lift the couch because it was too {d}.",
        "candidatos": ["man", "couch"],
        "variantes": {"weak": "man", "heavy": "couch"},
    },
    {
        "nombre": "plane_runway",
        "plantilla": "The plane couldn't land on the runway because it was too {d}.",
        "candidatos": ["plane", "runway"],
        "variantes": {"heavy": "plane", "short": "runway"},
    },
    {
        "nombre": "car_truck",
        "plantilla": "The car couldn't pass the truck because it was too {d}.",
        "candidatos": ["car", "truck"],
        "variantes": {"slow": "car", "wide": "truck"},
    },
    {
        "nombre": "backpack_locker",
        "plantilla": "The backpack doesn't fit in the locker because it is too {d}.",
        "candidatos": ["backpack", "locker"],
        "variantes": {"big": "backpack", "small": "locker"},
    },
]

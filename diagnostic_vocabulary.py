"""The 20 chord kinds and five intervallic families used in the article."""


VOCABULARY = [
    ("C", (0,0,0,1,0,0,1,0,0,0,0), "Major-third"),
    ("C6", (0,0,0,1,0,0,1,0,1,0,0), "Major-third"),
    ("C69", (0,1,0,1,0,0,1,0,1,0,0), "Major-third"),
    ("Cmaj7", (0,0,0,1,0,0,1,0,0,0,1), "Major-third"),
    ("Cmaj9", (0,1,0,1,0,0,1,0,0,0,1), "Major-third"),
    ("C7", (0,0,0,1,0,0,1,0,0,1,0), "Major-third"),
    ("C9", (0,1,0,1,0,0,1,0,0,1,0), "Major-third"),
    ("Cmi", (0,0,1,0,0,0,1,0,0,0,0), "Minor-third"),
    ("Cmi6", (0,0,1,0,0,0,1,0,1,0,0), "Minor-third"),
    ("Cmi7", (0,0,1,0,0,0,1,0,0,1,0), "Minor-third"),
    ("Cmi9", (0,1,1,0,0,0,1,0,0,1,0), "Minor-third"),
    ("Cmi69", (0,1,1,0,0,0,1,0,1,0,0), "Minor-third"),
    ("C7sus2", (0,1,0,0,0,0,1,0,0,1,0), "Suspended"),
    ("C7sus4", (0,0,0,0,1,0,1,0,0,1,0), "Suspended"),
    ("C7sus2sus4", (0,1,0,0,1,0,1,0,0,1,0), "Suspended"),
    ("Cdim", (0,0,1,0,0,1,0,0,0,0,0), "Diminished"),
    ("Cdim7", (0,0,1,0,0,1,0,0,1,0,0), "Diminished"),
    ("Cmi7b5", (0,0,1,0,0,1,0,0,0,1,0), "Diminished"),
    ("Caug", (0,0,0,1,0,0,0,1,0,0,0), "Augmented"),
    ("Caug7", (0,0,0,1,0,0,0,1,0,1,0), "Augmented"),
]

FAMILY_ORDER = [
    "Major-third",
    "Minor-third",
    "Suspended",
    "Diminished",
    "Augmented",
]

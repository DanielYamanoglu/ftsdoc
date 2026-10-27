ENTITY_TYPES = ["PERSON", "ORG", "DATE", "MONEY", "EMAIL", "SKILL", "INVOICE_NO", "PART_NO"]
LABELS = ["O"] + [f"{p}-{t}" for t in ENTITY_TYPES for p in ("B", "I")]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

demo1 = {
    "text": "Ad Notes",
    "output": "[ ( NP-HLN ( NN Ad ) ( NNS Notes ) ) ]",
}


demo2 = {
    "text": "The market crumbled",
    "output": "[ ( S ( NP-SBJ ( DT The ) ( NN market ) ) ( VP ( VBD crumbled ) ) ) ]",
}

demo3 = {
    "text": "I felt betrayed he later said.",
    "output": "[ ( S ( S-TPC-1 ( NP-SBJ ( PRP I ) ) ( VP ( VBD felt ) ( ADJP-PRD ( VBN betrayed ) ) ) ) ( NP-SBJ ( PRP he ) ) ( ADVP-TMP ( RB later ) ) ( VP ( VBD said ) ) ) ]",
}

demo4 = {
    "text": "Friday October 13 1989",
    "output": "[ ( NP ( NNP Friday ) ( NNP October ) ( CD 13 ) ( CD 1989 ) ) ]",
}

demo5 = {
    "text": "The Arabs had merely oil",
    "output": "[ ( S ( NP-SBJ ( DT The ) ( NNPS Arabs ) ) ( VP ( VBD had ) ( NP ( RB merely ) ( NN oil ) ) ) ) ]",
}


demo6 = {
    "text": "Energy",
    "output": "[ ( NP-HLN ( NN Energy ) ) ]",
}

demo7 = {
    "text": "Some U.S. entrepreneurs operate on a smaller scale",
    "output": "[ ( S ( NP-SBJ ( DT Some ) ( NNP U.S. ) ( NNS entrepreneurs ) ) ( VP ( VBP operate ) ( PP-MNR ( IN on ) ( NP ( DT a ) ( JJR smaller ) ( NN scale ) ) ) ) ) ]",
}

demo8 = {
    "text": "Knowledgeware Inc.",
    "output": "[ ( NP-HLN ( NNP Knowledgeware ) ( NNP Inc. ) ) ]",
}

demo9 = {
    "text": "And her husband sometimes calls her Ducky",
    "output": "[ ( S ( CC And ) ( NP-SBJ ( PRP$ her ) ( NN husband ) ) ( ADVP-TMP ( RB sometimes ) ) ( VP ( VBZ calls ) ( S ( NP-SBJ ( PRP her ) ) ( NP-PRD ( NNP Ducky ) ) ) ) ) ]",
}

demo10 = {
    "text": "Nausea seems a commonplace symptom",
    "output": "[ ( S ( NP-SBJ ( NN Nausea ) ) ( VP ( VBZ seems ) ( NP-PRD ( DT a ) ( JJ commonplace ) ( NN symptom ) ) ) ) ]",
}

demo11 = {
    "text": "Swedish Export Credit Corp Sweden",
    "output": "[ ( NP ( NP ( NNP Swedish ) ( NNP Export ) ( NNP Credit ) ( NNP Corp ) ) ( PRN ( NP-LOC ( NNP Sweden ) ) ) ) ]",
}

demo12 = {
    "text": "The dollar weakened against most other major currencies",
    "output": "[ ( S ( NP-SBJ ( DT The ) ( NN dollar ) ) ( VP ( VBD weakened ) ( PP ( IN against ) ( NP ( RBS most ) ( JJ other ) ( JJ major ) ( NNS currencies ) ) ) ) ) ]",
}


DEMOs = [
    demo1,
    demo2,
    demo3,
    demo4,
    demo5,
    demo6,
    demo7,
    demo8,
    demo9,
    demo10,
    demo11,
    demo12,
]

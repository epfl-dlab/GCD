import re

# Taken from train split

demo1 = {
    "text": "Iraqi Kurdish areas bordering Iran are under the control of guerrillas of the Iraqi Kurdish Patriotic Union"
    " of Kurdistan [START_ENT] PUK [END_ENT] group Puk and Iraq s Kurdistan Democratic Party Kdp the two main Iraqi "
    "Kurdish factions have had northern Iraq under their control since Iraqi forces were ousted from Kuwait in the 1991 Gulf War Clashes ",
    "output": "PUK [ Patriotic Union of Kurdistan ]",
}

demo2 = {
    "text": "Eu rejects [START_ENT] German [END_ENT] call to boycott British lamb Peter Blackburn Brussels 1996 08 22 The European Commission said on Thursday",
    "output": "German [ Germany ]",
}

demo3 = {
    "text": "Eu rejects German call to boycott [START_ENT] British [END_ENT] lamb Peter Blackburn Brussels 1996 08 22 The European Commission",
    "output": "British [ United Kingdom ]",
}

demo4 = {
    "text": "guitar legend Jimi Hendrix was sold for almost 17 000 on Thursday at an auction of some of the late musician "
    "s favourite possessions A Florida restaurant paid 10 925 pounds 16 935 for the draft of Ai n t no telling which "
    "Hendrix penned on a piece of London hotel stationery in late 1966 At the end of a January 1967 concert in the "
    "English city of Nottingham he threw the sheet of paper into the audience where it was retrieved by a fan Buyers "
    "also snapped up 16 other items that were put up for auction by [START_ENT] Hendrix [END_ENT] s former girlfriend Kathy Etchingham",
    "output": "Hendrix [ Jimi Hendrix ]",
}

demo5 = {
    "text": "Almost all German car manufacturers posted gains in registration numbers in the period [START_ENT] Volkswagen AG "
    "[END_ENT] won 77 719 registrations slightly more than a quarter of the total Opel Ag together with General Motors",
    "output": "Volkswagen AG [ Volkswagen ]",
}

demo6 = {
    "text": "The Greek socialist party s executive bureau gave the green light to Prime Minister Costas Simitis to call "
    "snap elections its general secretary [START_ENT] Costas Skandalidis [END_ENT] told reporters Prime Minister "
    "Costas Simitis is going to make an official announcement",
    "output": "Costas Skandalidis [ Kostas Skandalidis ]",
}

demo7 = {
    "text": "The following bond was announced by lead manager Toronto Dominion Borrower [START_ENT] Bayerische Vereinsbank"
    " [END_ENT] Amt C 100 Mln Coupon 6 625 Maturity 24 Sep 02 ",
    "output": "Bayerische Vereinsbank [ HypoVereinsbank ]",
}


DEMOs = [demo1, demo2, demo3, demo4, demo5, demo6, demo7]

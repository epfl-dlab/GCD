import re

# Taken from train split

demo1 = {
    "text": "of guerrillas of the Iraqi Kurdish Patriotic Union of Kurdistan [START_ENT] PUK [END_ENT] group Puk and Iraq s Kurdistan Democratic Party Kdp the",
    "output": "PUK : Canonical Form [ Patriotic Union of Kurdistan ]",
}

demo2 = {
    "text": "Eu rejects [START_ENT] German [END_ENT] call to boycott British lamb Peter Blackburn Brussels 1996 08",
    "output": "German : Canonical Form [ Germany ]",
}

demo3 = {
    "text": "Eu rejects German call to boycott [START_ENT] British [END_ENT] lamb Peter Blackburn Brussels 1996 08 22 The European Commission",
    "output": "British : Canonical Form [ United Kingdom ]",
}

demo4 = {
    "text": "16 other items that were put up for auction by [START_ENT] Hendrix [END_ENT] s former girlfriend Kathy Etchingham",
    "output": "Hendrix : Canonical Form [ Jimi Hendrix ]",
}

demo5 = {
    "text": "German car manufacturers posted gains in registration numbers in the period [START_ENT] Volkswagen AG "
    "[END_ENT] won 77 719 registrations slightly more than a quarter of the ",
    "output": "Volkswagen AG : Canonical Form [ Volkswagen ]",
}

demo6 = {
    "text": "Prime Minister Costas Simitis to call snap elections its general secretary [START_ENT] Costas Skandalidis [END_ENT] told reporters Prime Minister "
    "Costas Simitis is going to make an official announcement",
    "output": "Costas Skandalidis : Canonical Form [ Kostas Skandalidis ]",
}

demo7 = {
    "text": "The following bond was announced by lead manager Toronto Dominion Borrower [START_ENT] Bayerische Vereinsbank"
    " [END_ENT] Amt C 100 Mln Coupon 6 625 Maturity 24 Sep 02 ",
    "output": "Bayerische Vereinsbank : Canonical Form [ HypoVereinsbank ]",
}


DEMOs = [demo1, demo2, demo3, demo4, demo5, demo6, demo7]

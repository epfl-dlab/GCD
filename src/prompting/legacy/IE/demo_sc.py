#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : demo_sc.py
# @Date : 2023-04-28-14-39
# @Project: SynthIE
# @AUTHOR : Saibo Geng
# @Desc :

demo1 = {
    "text": "Vettaikaaran (2009 film) was originally written in the Tamil language, with B. Babusivan as the "
    "screenwriter.",
    "output": "[s] Vettaikaaran_(2009_film) [r] original language of film or TV show [o] Tamil_language [r] "
    "screenwriter [o] B._Babusivan [e] ",
}

demo2 = {
    "text": "The NHL Stadium Series is a sport that consists of ice hockey.",
    "output": "[s] NHL_Stadium_Series [r] sport [o] Ice_hockey [e]",
}

demo3 = {
    "text": "Abhishek Pictures is a film production company based in Hyderabad.",
    "output": "[s] Abhishek_Pictures [r] industry [o] Film_industry [r] headquarters location [o] Hyderabad [e]",
}

demo4 = {
    "text": "Swedish Open Cultural Heritage is a project developed by the Swedish National Heritage Board, "
    "which is mainly focused on cultural heritage. It produces Resource Description Framework as its product "
    "or material and uses XML, JSON, and JSON-LD as its file formats. XML was inspired by Standard "
    "Generalized Markup Language.",
    "output": "[s] Swedish_Open_Cultural_Heritage [r] main subject [o] Cultural_heritage [r] developer [o] "
    "Swedish_National_Heritage_Board [r] product or material produced [o] Resource_Description_Framework [r] "
    "file format [o] XML [r] file format [o] JSON [r] file format [o] JSON-LD [e] [s] XML [r] inspired by ["
    "o] Standard_Generalized_Markup_Language [e]",
}

demo5 = {
    "text": "The General Administration of Quality Supervision, Inspection and Quarantine was replaced by the State "
    "Administration for Market Regulation and is a government agency under the parent organization, "
    "the State Council of the People's Republic of China. Its headquarters is located in Haidian District, "
    "China.",
    "output": "[s] General_Administration_of_Quality_Supervision,_Inspection_and_Quarantine [r] replaced by [o] "
    "State_Administration_for_Market_Regulation [r] instance of [o] Government_agency [r] parent "
    "organization [o] State_Council_of_the_People's_Republic_of_China [r] country [o] China [r] "
    "headquarters location [o] Haidian_District [e]",
}

demo6 = {
    "text": "Marcus Jacob Monrad was a Lutheran who passed away in Oslo and was buried in the Cemetery of Our Saviour.Marcus Jacob Monrad was a Lutheran who passed away in Oslo and was buried in the Cemetery of Our Saviour.",
    "output": "[s] Marcus_Jacob_Monrad [r] place of death [o] object  [r] place of burial [o] "
    "Cemetery_of_Our_SaviourCemetery_of_Our_Saviour [r] religion [o] Lutheranism [e]",
}

demo7 = {
    "text": "The Making of Maddalena was shot by James Van Trees as director of photography and is presented in black and white. The cast includes Edna Goodrich.",
    "output": "[s] The_Making_of_Maddalena [r] director of photography [o] James_Van_Trees [r] color [o] black_and_white [r] cast member [o] Edna_Goodrich [e]",
}

demo8 = {
    "text": "The Artemis Accords are a set of agreements created by NASA and valid in outer space, with Australia, Ukraine, and Colombia as signatories. Outer space is used for spaceflight, which is the opposite of astronomical objects.The Artemis Accords are a set of agreements created by NASA and valid in outer space, with Australia, Ukraine, and Colombia as signatories. Outer space is used for spaceflight, which is the opposite of astronomical objects.",
    "output": "[s] Artemis_Accords [r] creator [o] NASA [r] valid in place [o] Outer_space [r] signatory [o] Australia [r] signatory [o] Ukraine [r] signatory [o] Colombia [e] [s] outer_space [r] use [o] Spaceflight [r] opposite of [o] Astronomical_object [e]",
}

demo9 = {
    "text": "Papaver rhoeas is a species of plant that is invasive to China and has a variety of uses, such as the production of fruit.",
    "output": "[s] Papaver_rhoeas [r] taxon rank [o] Species [r] invasive to [o] China [r] use [o] Fruit [r] instance of [o] Taxon [e]",
}

demo10 = {
    "text": "Gymnastics at the 2006 Commonwealth Games was a sport held in Australia as part of the 2006 Commonwealth Games, an instance of Season (sports). Australia has a diplomatic relation with Mauritius.",
    "output": "[s] Gymnastics_at_the_2006_Commonwealth_Games [r] sport [o] Gymnastics [r] part of [o] 2006_Commonwealth_Games [r] country [o] Australia [r] instance of [o] Season_(sports) [e] [s] Australia [r] diplomatic relation [o] Mauritius [e]",
}

demo11 = {
    "text": "Archibald Campbell, 2nd Earl of Argyll was a noble title held by Archibald Campbell, according to Nordisk familjebok. He belonged to the Clan Campbell family.",
    "output": "[s] Archibald_Campbell,_2nd_Earl_of_Argyll [r] noble title [o] Earl [r] described by source [o] Nordisk_familjebok [r] family [o] Clan_Campbell [e]",
}

demo12 = {
    "text": "Arena Zagreb is occupied by RK Zagreb and owned by Zagreb. It hosts a variety of sports, including tennis, and was the site of a groundbreaking event.",
    "output": "[s] Arena_Zagreb [r] owned by [o] Zagreb [r] occupant [o] RK_Zagreb [r] sport [o] Tennis [r] significant event [o] groundbreaking [e]",
}

demo13 = {
    "text": "An accelerometer is a measuring instrument that measures acceleration, which is a physical quantity that follows velocity. Pierre Varignon is credited as the discoverer or inventor of acceleration. The Brockhaus and Efron Encyclopedic Dictionary describes acceleration.",
    "output": "[s] Accelerometer [r] subclass of [o] Measuring_instrument [r] measures [o] Acceleration [e] [s] Acceleration [r] subclass of [o] Physical_quantity [r] follows [o] Velocity [r] discoverer or inventor [o] Pierre_Varignon [r] described by source [o] Brockhaus_and_Efron_Encyclopedic_Dictionary [e]",
}

demo14 = {
    "text": "Kensington and Chelsea Tenant Management Organisation (Kensington and Chelsea TMO) is a Limited company, located in Kensington, with Emma Dent Coad as one of its board members.",
    "output": "[s] Kensington_and_Chelsea_TMO [r] located in the administrative territorial entity [o] Kensington [r] legal form [o] Limited_company [r] board member [o] Emma_Dent_Coad [e]",
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
    demo13,
    demo14,
]

# num_demo=13 => overhead_tokens = 1601
# num_demo=14 => overhead_tokens = 1711

# max = 2048-256 = 1792

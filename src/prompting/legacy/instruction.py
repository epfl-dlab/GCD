#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : instruction.py
# @Date : 2023-05-27-21-50
# @Project: SynthIE
# @AUTHOR : Saibo Geng
# @Desc :

INSTRUCTIONS = {
    "el": {
        "basic": "Disambiguate the entity surrounded by [START_ENT] and [END_ENT] by giving the correct entity name in the knowledge base: \n",
        "canonical": "Disambiguate the entity surrounded by [START_ENT] and [END_ENT] by giving the canonical entity name in the knowledge base: \n",
    },
    "ie": {
        "fe": "Extract the triples in fully-expanded format from texts below. \n",
        "sc": "Extract the triples in subject-collapsed format from texts below. \n",
        "feR": "Extract the triples in fully-expanded format from texts below. The relation should be in this list: patron saint;killed by;partner in business or sport;operator;described by source;cover art by;owned by;film editor;librettist;illustrator;twinned administrative body;located in the administrative territorial entity;performer;director of photography;adjacent station;home port;film crew member;participant in;highway system;country of origin;location of formation;chief executive officer;mother house;followed by;record label;notable work;filming location;basin country;board member;connecting service;location of creation;production company;original language of film or TV show;characters;award received;director;culture;part of;religious order;main building contractor;inflows;capital of;nominated for;carries;place of publication;place of detention;located in or next to body of water;winner;noble title;located on terrain feature;architect;affiliation;narrative location;located on astronomical body;location;doctoral advisor;screenwriter;main subject;movement;has part;creator;executive producer;employer;exclave of;occupation;spouse;voice actor;place of origin (Switzerland);relative;follows;member of;capital;author;member of political party;lake outflow;conflict;influenced by;located in protected area;named after;located on linear feature;founded by;publisher;participant;located on street;subclass of;subsidiary;sport;languages spoken, written or signed;after a work by;time period;canonization status;cast member;maintained by;military branch;producer;country of citizenship;lyrics by;student of;commander of (DEPRECATED);crosses;student;narrator;presenter;original broadcaster;chairperson;cause of death;drafted by;residence;religion;part of the series;shares border with;headquarters location;place of burial;country;contains settlement;composer;unmarried partner;mouth of the watercourse;terminus location;father;country for sport;based on;architectural style;origin of the watercourse;place served by transport hub;head of government;doctoral student;lowest point;child;parent organization;broadcast by;recorded at studio or venue;stock exchange;manufacturer;heritage designation;place of birth;place of death;dedicated to;continent;work location;member of sports team;sibling;allegiance;distributed by;head coach;backup or reserve team or crew;archives at;production designer;designed by;diocese;position held;connecting line;genre;educated at;occupant;language of work or name;family;airline hub\n",
        "scR": "Extract the triples in subject-collapsed format from texts below. The relation should be in this list: patron saint;killed by;partner in business or sport;operator;described by source;cover art by;owned by;film editor;librettist;illustrator;twinned administrative body;located in the administrative territorial entity;performer;director of photography;adjacent station;home port;film crew member;participant in;highway system;country of origin;location of formation;chief executive officer;mother house;followed by;record label;notable work;filming location;basin country;board member;connecting service;location of creation;production company;original language of film or TV show;characters;award received;director;culture;part of;religious order;main building contractor;inflows;capital of;nominated for;carries;place of publication;place of detention;located in or next to body of water;winner;noble title;located on terrain feature;architect;affiliation;narrative location;located on astronomical body;location;doctoral advisor;screenwriter;main subject;movement;has part;creator;executive producer;employer;exclave of;occupation;spouse;voice actor;place of origin (Switzerland);relative;follows;member of;capital;author;member of political party;lake outflow;conflict;influenced by;located in protected area;named after;located on linear feature;founded by;publisher;participant;located on street;subclass of;subsidiary;sport;languages spoken, written or signed;after a work by;time period;canonization status;cast member;maintained by;military branch;producer;country of citizenship;lyrics by;student of;commander of (DEPRECATED);crosses;student;narrator;presenter;original broadcaster;chairperson;cause of death;drafted by;residence;religion;part of the series;shares border with;headquarters location;place of burial;country;contains settlement;composer;unmarried partner;mouth of the watercourse;terminus location;father;country for sport;based on;architectural style;origin of the watercourse;place served by transport hub;head of government;doctoral student;lowest point;child;parent organization;broadcast by;recorded at studio or venue;stock exchange;manufacturer;heritage designation;place of birth;place of death;dedicated to;continent;work location;member of sports team;sibling;allegiance;distributed by;head coach;backup or reserve team or crew;archives at;production designer;designed by;diocese;position held;connecting line;genre;educated at;occupant;language of work or name;family;airline hub\n",
    },
    "cp": {
        "ptb": "Perform constituency parsing on the provided sentences in accordance with the Penn TreeBank annotation guidelines. \n",
        "ptbCoT": "Perform constituency parsing on the provided sentences in accordance with the Penn TreeBank annotation guidelines. "
        "Let's look at your example: 'Mr. Roberts was assistant Treasury secretary under President Reagan'. "
        "The constituency parse tree is:  "
        "( S ( NP-SBJ ( NNP Mr. ) ( NNP Roberts ) ) ( VP ( VBD was ) ( NP-PRD ( JJ assistant ) ( NNP Treasury ) ( NN secretary ) ) ( PP-LOC ( IN under ) ( NP ( NNP President ) ( NNP Reagan ) ) ) ) )"
        "S: This stands for Sentence, the top-level structure in the parse tree."
        "NP-SBJ: This is the subject noun phrase of the sentence. In this case, it's 'Mr. Roberts'."
        "VP: This stands for Verb Phrase, which is the part of the sentence that contains the verb and its direct and/or indirect objects, along with any other information. In this case, it's 'was assistant Treasury secretary under President Reagan'."
        "VBD: This stands for Verb, Past Tense. The word 'was' falls into this category."
        "NP-PRD: This is the predicate noun phrase. Here, it is 'assistant Treasury secretary'."
        "JJ: This stands for Adjective. The word 'assistant' falls into this category."
        "NNP: This stands for Proper Noun, Singular. 'Mr.', 'Roberts', 'Treasury', 'President', and 'Reagan' are all examples of this."
        "NN: This stands for Noun, Singular or Mass. The word 'secretary' falls into this category."
        "PP-LOC: This is a Prepositional Phrase that indicates location or direction. 'under President Reagan' falls into this category."
        "IN: This stands for Preposition or Subordinating Conjunction. The word 'under' falls into this category."
        "Now let's look at another example: 'Immune Response Corp. Three million common shares via Merrill Lynch'."
        "The constituency parse tree is: ( NP ( NP ( NNP Immune ) ( NNP Response ) ( NNP Corp. ) ) ( NP ( QP ( CD Three ) ( CD million ) ) ( JJ common ) ( NNS shares ) ) ( PP ( IN via ) ( NP ( NNP Merrill ) ( NNP Lynch ) ) ) )"
        "NP: This stands for Noun Phrase. The sentence is composed of multiple nested Noun Phrases."
        "NNP: This stands for Proper Noun, Singular. 'Immune', 'Response', 'Corp.', 'Merrill', and 'Lynch' are all examples of this."
        "QP: This is a Quantifier Phrase, used to specify the quantity of something. Here, it is 'Three million'."
        "CD: This stands for Cardinal Number, representing numbers that can be added together. 'Three' and 'million' fall into this category."
        "JJ: This stands for Adjective. The word 'common' falls into this category."
        "NNS: This stands for Noun, Plural. The word 'shares' falls into this category."
        "PP: This stands for Prepositional Phrase, a modifying phrase consisting of a preposition and its object. Here, it is 'via Merrill Lynch'."
        "IN: This stands for Preposition or Subordinating Conjunction. The word 'via' falls into this category.'"
        "Now let's look at another example: 'Bond prices were barely higher'."
        "The constituency parse tree is: ( S ( NP-SBJ ( NN Bond ) ( NNS prices ) ) ( VP ( VBD were ) ( ADJP-PRD ( RB barely ) ( JJR higher ) ) ) )"
        "S: This stands for Sentence, the top-level structure in the parse tree."
        "NP-SBJ: This is the subject noun phrase of the sentence, which is 'Bond prices'."
        "NN: This stands for Noun, Singular or Mass. In this case, the word 'Bond' falls into this category."
        "NNS: This stands for Noun, Plural. The word 'prices' is an example of this."
        "VP: This stands for Verb Phrase, which in this case is 'were barely higher'."
        "VBD: This stands for Verb, Past Tense. The word 'were' falls into this category."
        "ADJP-PRD: This stands for Adjective Phrase, used as a predicate. The phrase 'barely higher' is an example of this."
        "RB: This stands for Adverb. The word 'barely' falls into this category."
        "JJR: This stands for Adjective, Comparative. The word 'higher' falls into this category.'"
        "Now let's look at some other examples:",
    },
}

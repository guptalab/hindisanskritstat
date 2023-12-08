IndexToVyanjan = {
    0: b'\xe0\xa4\x85',
    1: b'\xe0\xa4\x95',
    2: b'\xe0\xa4\x96',
    3: b'\xe0\xa4\x97',
    4: b'\xe0\xa4\x98',
    5: b'\xe0\xa4\x99',
    6: b'\xe0\xa4\x9a',
    7: b'\xe0\xa4\x9b',
    8: b'\xe0\xa4\x9c',
    9: b'\xe0\xa4\x9d',    
    10: b'\xe0\xa4\x9e',
    11: b'\xe0\xa4\x9f',
    12: b'\xe0\xa4\xa0',
    13: b'\xe0\xa4\xa1',
    14: b'\xe0\xa4\xa2',
    15: b'\xe0\xa4\xa3',
    16: b'\xe0\xa4\xa4',
    17: b'\xe0\xa4\xa5',
    18: b'\xe0\xa4\xa6',
    19: b'\xe0\xa4\xa7',
    20: b'\xe0\xa4\xa8',
    21: b'\xe0\xa4\xaa',
    22: b'\xe0\xa4\xab',
    23: b'\xe0\xa4\xac',
    24: b'\xe0\xa4\xad',
    25: b'\xe0\xa4\xae',
    26: b'\xe0\xa4\xaf',
    27: b'\xe0\xa4\xb0',
    28: b'\xe0\xa4\xb2', 
    29: b'\xe0\xa4\xb5',
    30: b'\xe0\xa4\xb6',
    31: b'\xe0\xa4\xb7',
    32: b'\xe0\xa4\xb8',
    33: b'\xe0\xa4\xb9',   
}

LetterToIndex ={
    b'\xe0\xa4\x95': 1,
    b'\xe0\xa4\x96': 2,
    b'\xe0\xa4\x97': 3,
    b'\xe0\xa4\x98': 4,
    b'\xe0\xa4\x99': 5,
    b'\xe0\xa4\x9a': 6,
    b'\xe0\xa4\x9b': 7,
    b'\xe0\xa4\x9c': 8,
    b'\xe0\xa4\x9d': 9, 
    b'\xe0\xa4\x9e': 10,
    b'\xe0\xa4\x9f': 11,
    b'\xe0\xa4\xa0': 12,
    b'\xe0\xa4\xa1': 13,
    b'\xe0\xa4\xa2': 14,
    b'\xe0\xa4\xa3': 15,
    b'\xe0\xa4\xa4': 16,
    b'\xe0\xa4\xa5': 17,
    b'\xe0\xa4\xa6': 18,
    b'\xe0\xa4\xa7': 19,
    b'\xe0\xa4\xa8': 20,
    b'\xe0\xa4\xaa': 21,
    b'\xe0\xa4\xab': 22,
    b'\xe0\xa4\xac': 23,
    b'\xe0\xa4\xad': 24,
    b'\xe0\xa4\xae': 25,
    b'\xe0\xa4\xaf': 26,
    b'\xe0\xa4\xb0': 27,
    b'\xe0\xa4\xb2': 28, 
    b'\xe0\xa4\xb5': 29,
    b'\xe0\xa4\xb6': 30,
    b'\xe0\xa4\xb7': 31,
    b'\xe0\xa4\xb8': 32,
    b'\xe0\xa4\xb9': 33,    
}


LetterType = {
    'क': 'vyanjan', 'ख': 'vyanjan', 'ग': 'vyanjan', 'घ': 'vyanjan', 'ङ': 'vyanjan',
    'च': 'vyanjan', 'छ': 'vyanjan', 'ज': 'vyanjan', 'झ': 'vyanjan', 'ञ': 'vyanjan',
    'ट': 'vyanjan', 'ठ': 'vyanjan', 'ड': 'vyanjan', 'ढ': 'vyanjan', 'ण': 'vyanjan',
    'त': 'vyanjan', 'थ': 'vyanjan', 'द': 'vyanjan', 'ध': 'vyanjan', 'न': 'vyanjan',
    'प': 'vyanjan', 'फ': 'vyanjan', 'ब': 'vyanjan', 'भ': 'vyanjan', 'म': 'vyanjan',
    'य': 'vyanjan', 'र': 'vyanjan', 'ल': 'vyanjan', 'व': 'vyanjan', 'श': 'vyanjan',
    'ष': 'vyanjan', 'स': 'vyanjan', 'ह': 'vyanjan', 
    'अ': 'swar', 'आ': 'swar', 'इ': 'swar', 'ई': 'swar', 'उ': 'swar', 'ऊ': 'swar', 'ए': 'swar', 'ऐ': 'swar', 'ओ': 'swar', 'औ': 'swar', 'अं': 'swar', 'अः': 'swar',
    'ा': 'matra', 'ि': 'matra', 'ी': 'matra', 'ु': 'matra', 'ू': 'matra', 'े': 'matra', 'ै': 'matra', 'ो': 'matra', 'ौ': 'matra', 'ं': 'matra', 'ः': 'matra',  
    '।': 'viram', '्': 'halant', '॥': 'danda', 'ँ': 'matra', 'ृ': 'matra','०': 'number', '१': 'number', '२': 'number', '३': 'number', '४': 'number', '५': 'number', '६': 'number', '७': 'number', '८': 'number', '९': 'number', 'ॐ': 'om'
    
}

vyanjans = ['क', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'झ', 'ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न', 'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ल', 'व', 'श', 'ष', 'स', 'ह']

swars = ['अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ए', 'ऐ', 'ओ', 'औ', 'अं', 'अः','ऋ', 'ऽ']

matras = ['ा', 'ि', 'ी', 'ु', 'ू', 'े', 'ै', 'ो', 'ौ', 'ं', 'ः', 'ँ', 'ृ']

avagrahas = ['ऽ']

specials = ['।', '्', '॥', 'ँ', 'ृ','०', '१', '२', '३', '४', '५', '६', '७', '८', '९', 'ॐ']

IndexToSwar = {
    1:'अ',
    2:'आ',
    3:'इ',
    4:'ई',
    5:'उ',
    6:'ऊ',
    7:'ए',
    8:'ऐ',
    9:'ओ',
    10:'औ',
    11:'अं',
    12:'अः',
}

IndexToMatra = {
    1: 'ा',
    2: 'ि',
    3: 'ी',
    4: 'ु',
    5: 'ू',
    6: 'े',
    7: 'ै',
    8: 'ो',
    9: 'ौ',
    10: 'ं',
    11: 'ः',
    12: 'ँ',
    13: 'ृ',
    
}

MatraToIndex = {
    
}

VyanjanToIndex = {
    'क': 1, 'ख': 2, 'ग': 3, 'घ': 4, 'ङ': 5, 
    'च': 6, 'छ': 7, 'ज': 8, 'झ': 9, 'ञ': 10,
    'ट': 11, 'ठ': 12, 'ड': 13, 'ढ': 14, 'ण': 15,
    'त': 16, 'थ': 17, 'द': 18, 'ध': 19, 'न': 20,
    'प': 21, 'फ': 22, 'ब': 23, 'भ': 24, 'म': 25,
    'य': 26, 'र': 27, 'ल': 28, 'व': 29, 'श': 30,
    'ष': 31, 'स': 32, 'ह': 33,
}

        
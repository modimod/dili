from collections import OrderedDict

tasks = OrderedDict([
		('sakatis', 1), ('zhu', 1), ('xu', 1), ('greene', 3), ('vnctr', 4), ('nctr', 3), ('severity', 9)])

tasks_idx = dict([('sakatis', 0), ('zhu', 1), ('xu', 2), ('greene', 3), ('vnctr', 4), ('nctr', 5), ('severity', 6)])

pandas_cols = ['Sakatis', 'Zhu', 'Xu', 'Greene', 'vNCTR', 'NCTR', 'Severity Class']

smiles_alphabet = '#%()+-./0123456789=@ABCFGHIKLMNOPRSTVZ[\\]abcdeghilnorstu'
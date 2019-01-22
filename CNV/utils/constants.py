from collections import OrderedDict

tasks = OrderedDict([
		('sakatis', 1), ('zhu', 1), ('xu', 1), ('greene', 3), ('vnctr', 3), ('nctr', 3), ('severity', 9)])

tasks_rank = OrderedDict([
		('sakatis', 1), ('zhu', 1), ('xu', 1), ('greene', 2), ('vnctr', 2), ('nctr', 2), ('severity', 8)])

tasks_label_count = OrderedDict([
		('sakatis', 2), ('zhu', 2), ('xu', 2), ('greene', 3), ('vnctr', 3), ('nctr', 3), ('severity', 9)])


tasks_idx = dict([('sakatis', 0), ('zhu', 1), ('xu', 2), ('greene', 3), ('vnctr', 4), ('nctr', 5), ('severity', 6), ('DILI', 0)])

pandas_cols = ['Sakatis', 'Zhu', 'Xu', 'Greene', 'vNCTR', 'NCTR', 'Severity Class']

smiles_alphabet = '#%()+-./0123456789=@ABCFGHIKLMNOPRSTVZ[\\]abcdeghilnorstu'

descr_dim = 186



# tasks = OrderedDict([
# 		('sakatis', 1), ('zhu', 1), ('xu', 1), ('greene', 3), ('vnctr', 3), ('nctr', 3)])
#
# tasks_rank = OrderedDict([
# 		('sakatis', 1), ('zhu', 1), ('xu', 1), ('greene', 2), ('vnctr', 2), ('nctr', 2)])
#
# tasks_label_count = OrderedDict([
# 		('sakatis', 2), ('zhu', 2), ('xu', 2), ('greene', 3), ('vnctr', 3), ('nctr', 3)])
#
# tasks_idx = dict([('sakatis', 0), ('zhu', 1), ('xu', 2), ('greene', 3), ('vnctr', 4), ('nctr', 5), ('DILI', 0)])
#
#
# pandas_cols = ['Sakatis', 'Zhu', 'Xu', 'Greene', 'vNCTR', 'NCTR', 'Severity Class']

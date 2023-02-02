import math
import os

# ----------- MAIN TESTS ----------

concepts_to_map = {  # high level - trivial - non trivial
    'Commercial': ['Restaurant', 'Billboard', 'Statue'],
    'Residential': ['MiscResidential', 'Porch', 'TiledRoof'],
    'Industrial': ['MiscIndustrial', 'Machine', 'Chimney'],
}

for complexity in range(1, 7):
    for main_class in concepts_to_map:
        for concept in concepts_to_map[main_class]:

            extractor_path = './extractors_new/Ext_newC' + str(complexity) + '_' + main_class + '_new'
            command = "python train_search.py " \
                      ' --folder_name paper_experiments' \
                      ' --main_network ' + main_class + \
                      ' --concept ' + concept + \
                      ' --complexity ' + str(complexity)

            print('\n\n')
            print('-' * 160)
            print('{} {} {}'.format(complexity, main_class, concept))
            print(command)
            os.system('cmd /c "' + command + '"')

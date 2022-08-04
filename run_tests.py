import math
import os

for layer in [1, 2, 3, 4]:
    for extractor_path in ['./extractors/xtrains_typeA', './extractors/xtrains_typeB', './extractors/xtrains_typeC']:
        for label in ['FreightWagon', 'WarTrain', 'EmptyTrain', 'ReinforcedCar', 'PassengerTrain', 'LongTrain',
                      'FreightTrain', 'LongWagon', 'RuralTrain', 'MixedTrain', 'OpenRoofCar']:
            command = "python train_search.py --layers " + str(layer) + ' --extractor_path ' + extractor_path + ' --sec_labels ' + label + ' --init_channels ' + str(int(math.pow(2, layer) * 8))
            os.system('cmd /c "' + command + '"')

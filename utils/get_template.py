import pandas as pd
import os


def extract_template(dataset):
    # Data Cleaning extract useful token for prompt template
    if dataset == 'waterbird':
        # 1. style label
        style_label = ['photo']
        # 2. class label
        # class_label = ['standing bird','flying bird']
        class_label = ['bird','bird']
        # 3. background label
        # background_info = ['ocean','forest']
        background_info = ['ocean','lake','forest','bamboo forest']
        # 4. template
        template = ['a {} of {} in {} background']
        # 5. random adj for background token
        rand_adj = ['green','blue','beautiful','safe','comfortable','tough']
    elif dataset == 'celebA':
        style_label = ['photo']
        class_label = ['blond','non-blond']
        background_info = ['man','woman']
        template = ['a {} of a {} {}']
        rand_adj = [' ','handsome','beautiful','normal']
    return style_label, class_label, background_info, template,rand_adj




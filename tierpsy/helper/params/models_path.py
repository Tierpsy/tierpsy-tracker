from tierpsy import AUX_FILES_DIR
import os
import warnings


def get_model_filter_worms(p_dict):

    which_model = p_dict['nn_filter_to_use']
    if which_model != 'custom':
        if p_dict['path_to_custom_pytorch_model'] != '':
            warnings.warn('A path to a custom model wass provided, '
                          + f'but "nn_filter_to_use" was set to {which_model}.'
                          + ' The custom path will be ignored.')

    if which_model == 'tensorflow_default':
        model_filter_worms = os.path.join(AUX_FILES_DIR,
                                          'model_isworm_20170407_184845.h5')
    elif which_model == 'pytorch_default':
        model_filter_worms = os.path.join(AUX_FILES_DIR,
                                          'model_state_isworm_20200615.pth')
    elif which_model == 'custom':
        model_filter_worms = p_dict['path_to_custom_pytorch_model']
        if model_filter_worms == '':
            raise ValueError(
                'The path to the custom pytorch model to filter spurious '
                'particles was not given. Set "path_to_custom_pytorch_model" '
                'or select a different "nn_filter_to_use".'
            )
        if not os.path.exists(model_filter_worms):
            raise FileNotFoundError(
                'The custom pytorch model file to filter spurious particles '
                f'was not found: {model_filter_worms}'
            )
    elif which_model == 'none':
        model_filter_worms = ''
    else:
        raise Exception('Invalid option for model_filter_worms')

    if model_filter_worms != '':
        if not os.path.exists(model_filter_worms):
            warnings.warn('The selected model file to filter '
                          + 'spurious particles was not found. '
                          + 'This step will not be done.')
            model_filter_worms = ''

    return model_filter_worms


DFLT_MODEL_FOOD_CONTOUR = os.path.join(AUX_FILES_DIR,
                                       'unet_RMSprop-5-04999-0.3997.h5')
if not os.path.exists(DFLT_MODEL_FOOD_CONTOUR):
    warnings.warn("The default model to obtain the food contour was not found."
                  + " I'll' try a less accurate algorithm"
                  + " to calculate the contour.")
    DFLT_MODEL_FOOD_CONTOUR = ''
